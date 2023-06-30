import LinearAlgebra: dot, I, UniformScaling, norm, ldiv!, mul!
import IncompleteLU: ILUFactorization

export bicgstab, bicgstab!, STATUS, R_SHADOW_STRATEGY, converged, running, reset, reached_limit, initial, perturbated, random, enriched

@enum STATUS converged=0 running=1 reset=2 reached_limit=3
@enum R_SHADOW_STRATEGY initial=0 perturbated=1 random=2 enriched=3

mutable struct History{T <: Number, ITER <: Number}
  
  xs :: Vector{AbstractVector{T}}
  residuals :: Vector{T}
  nIters :: ITER
  ωs :: Vector{T}
  rhos :: Vector{T}
  αDivs :: Vector{T}
  ωDivs :: Vector{T}
  leadingCoeffs :: Vector{T}
  nResets :: ITER
  status :: STATUS

  History{T, ITER}() where {T <: Number, ITER <: Number} = new(
    Vector{Vector{T}}(), 
    Vector{T}(),
    zero(ITER),
    Vector{T}(),
    Vector{T}(),
    Vector{T}(),
    Vector{T}(),
    Vector{T}(),
    zero(ITER),
    running
  )
end

@inbounds function square_sum(vv :: AbstractVector{T}) :: T where {T <: Number}
  out = zero(T)
  for v in vv
    out = out + v^2
  end 
  out
end

@inbounds function init_history!(
    history :: History{T, ITER},
    debug :: Bool,
    nIter :: ITER,
    xIter :: AbstractVector{T},
    r :: AbstractVector{T},
    rho :: T,
    ω :: T
  ) where {T <: Number, ITER <: Number}

  if debug
    push!(history.xs, xIter)
    push!(history.residuals, norm(r, 2))
    push!(history.rhos, rho)
    push!(history.ωs, ω)
  end 
end

@inbounds function update_history!(
    history :: History{T, ITER},
    xIter :: AbstractVector{T},
    r :: AbstractVector{T},
    i :: ITER,
    ω :: T,
    rho :: T,
    αDiv :: T,
    ωDiv :: T,
    leadingCoeff :: T
  ) where {T <: Number, ITER <: Number}

  id = i + one(i)
  push!(history.xs, copy(xIter))
  push!(history.residuals, norm(r, 2))
  push!(history.ωs, ω)
  push!(history.rhos, rho)
  push!(history.αDivs, αDiv)
  push!(history.ωDivs, ωDiv)
  push!(history.leadingCoeffs, leadingCoeff)
end

@inbounds function observable_failure(
    observablePrecision :: T,
    ω :: T,
    rho :: T,
    αDiv :: T,
    ωDiv :: T,
    leadingCoeff :: T
  ) :: Bool where {T <: Number}

  ω < observablePrecision || rho < observablePrecision || αDiv < observablePrecision || ωDiv < observablePrecision || leadingCoeff < observablePrecision
end

@inbounds function get_r_shadow(
  r :: AbstractVector{T},
  b :: AbstractVector{T},
  rShadowStrategy :: R_SHADOW_STRATEGY
  ) where {T <: Number}

  if rShadowStrategy == initial 
    return copy(r)
  elseif rShadowStrategy == perturbated
    perturbation = rand(T, length(r))
    return r .+ (norm(r, 2) / norm(perturbation, 2)) .* perturbation 
  elseif rShadowStrategy == random
    return rand(T, length(r))
  end
  # enriched
  return r .+ (norm(r, 2) / norm(b, 2)) .* b
end

@inbounds function _bicgstab( 
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T},
    history :: History,
    nIter :: ITER,
    debug :: Bool,
    v :: AbstractVector{T},
    p :: AbstractVector{T},
    y :: AbstractVector{T},
    s :: AbstractVector{T},
    z :: AbstractVector{T};
    K :: Union{AbstractMatrix{T}, ILUFactorization{T, TID}, UniformScaling{Bool}} = I,
    ϵ :: T = convert(T, 10e-11),
    resetEvery :: ITER = 1002,
    rShadowStrategy :: R_SHADOW_STRATEGY = initial,
    observablePrecision :: T = 10e-4 
  ) where {T <: Number, TID <: Integer, ITER <: Number}

  r = Vector{Float64}(undef, length(xIter))
  r .= b .- mul!(r, A, xIter)
  r_shadow = get_r_shadow(r, b, rShadowStrategy)
  past_rho = α = β = ω = one(T)
  i = history.nIters
  
  t = r 
  rho = dot(r_shadow, r)
  init_history!(history, debug, nIter, xIter, r, rho, ω)
  while i <= nIter

    β = (rho / past_rho) * (α / ω) 
    p .= r .+ β .* (p .- ω .* v)
    ldiv!(y, K, p)
    mul!(v, A, y) 
    αDiv = dot(r_shadow, v)
    α = rho / αDiv
    s .= r .- α .* v
    ldiv!(z, K, s)
    mul!(t, A, z)
    ωDiv = dot(t, t)
    ω = dot(t, s) / ωDiv
    past_rho = rho
    rho = -ω * dot(r_shadow, t)
    xIter .= xIter .+ α .* y .+ ω .* z
    if i % resetEvery == resetEvery - one(resetEvery)
      r .= b .- mul!(r, A, xIter)
    else
      r .= s .- ω .* t 
    end
    i = i + one(ITER)
    leadingCoeff = last(p)
    if debug
      update_history!(history, xIter, r, i, ω, rho, αDiv, ωDiv, leadingCoeff)
    end
    if square_sum(r) < ϵ
      history.status = converged
      history.nIters = i
      return
    end
  end
  history.status = reached_limit
  history.nIters = nIter
end

function bicgstab(
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    kwargs...
  ) where {T <: Number}

  bicgstab!(zeros(T, length(b)), A, b; kwargs...)
end

@inbounds function bicgstab!(
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    perturbateX :: Bool = true,
    τ :: T = 0.001,
    nIter :: ITER = UInt16(1000),
    debug :: Bool = false,
    kwargs...
  ) where {T <: Number, ITER <: Number}

  history = History{T, ITER}()

  n = length(xIter)
  v = zeros(T, n)
  p = zeros(T, n)
  y = Vector{T}(undef, n)
  s = Vector{T}(undef, n)
  z = Vector{T}(undef, n)

  while is_bicgstab_continuing(history)
    
    if (perturbateX)
      perturbation = rand(T, length(xIter))
      xIter .= τ .* norm(xIter, 2) .*  perturbation / norm(perturbation, 2) 
    end

    _bicgstab(xIter, A, b, history, nIter, debug, v, p, y, s, z; kwargs...)
  end
  
  if debug
    return (xIter, history)
  end
  xIter
end

function is_bicgstab_continuing(
    history :: History
  ) :: Bool 
  history.status == reset || history.status == running
end

