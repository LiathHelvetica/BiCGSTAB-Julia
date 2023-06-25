#=

abstrakcja - structy i multiple dispatch

Option = Union{T, Nothing}
Option = Union{Some}

I wonder if these definitions are effective

ldiv! może efektywnie wykonać kroki związane z obliczeniem układu w krokach prekondycjonowania ale musi być specjalnego formatu - czy można z tego skorzystać? czy trzeba ręcznie efektywnie obliczać układ

jeśli okaże się, że istnieje narzut na MVM przez implementację macierzy w Julii (związany z transpozycją i sprytnym wykonaniem tej operacji) to trzeba będzie zaimplementować własną macierz rzadką

Warto będzie spróbować tego choć nie jest to popularne
https://gridap.github.io/SparseMatricesCSR.jl/stable/
https://github.com/gridap/SparseMatricesCSR.jl

PRZEPROWADZIŁEM TEST - Nawet jeśli kwargs nie jest wykorzystywany to zajmuje pamięć
Więc tracimy potencjalnie wektor pamięci na macierz jednostkową przy braku prekondycjonowania

Sprawdzić - czy ID matrix zajmuje jakiekolwiek miejsce czy jest "Sprytna?"
 - nie no oczywiście, że zajmuje miesce i czas - głupio, że w ogóle myślałem, że może być inaczej
 - A nie - możliwe, że ldiv!(I, ...) jest hiperefektywne
 - do sprawdzenia
 - jak w ogóle działa UniformScaling

Jak efektywnie przechować wynik faktoryzacji
A równocześnie skalowalnie

Czy SparseMatrix jest typu AbstractMatrix? - wygląda na to, że tak da się je przypisać
Aczkolwiek teoretycznie nie są ze sobą powiązane

Jak efektywnie obliczyć układ równań prekondycjonowania
Krylov.jl wykorzystuje jakąś metodę mulorldiv! (?)
IterativeSolvers.jl wykorzystuje ldiv!
Ale w tych metodach macierz może być dowolnego typu 
Może jeśli zadeklaruję typ jako unię to styknie - unię typów które implementują ldiv

Ok, ale jakim cudem w takim razie ldiv jest zdefiniowane dla struktury ILUFactorization
src -> application.jl -> ldiv! - nie eksportuje tej funkcji

Dopóki liczba typów w unii nie przekracza 4 to Julia generuje wyspecjalizowany kod (?????? - przetestować na llvm)
https://docs.julialang.org/en/v1/manual/types/#Type-Unions
 - tak to faktycznie tak działa - zrobiłem test

W SUMIE
Cholesky poprzez zwykłe indeksowanie może być efektywny jeśli select w kolumnie wykonywany jest jako binsearch 
To również utrzymuje się dla kompresji rzędowej
Więc w metodzie powinno się dać pójść z kompresjonowanymi rzędami

Czy istnieje sposób aby łatwo podejrzeć ciało metody

=#

import LinearAlgebra: dot, I, UniformScaling, norm, ldiv!, mul!
import IncompleteLU: ILUFactorization

export bicgstab, bicgstab!

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

function square_sum(vv :: AbstractVector{T}) :: T where {T <: Number}
  out = zero(T)
  for v in vv
    out = out + v^2
  end 
  out
end

function init_history!(
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
    push!(history.residuals, square_sum(r))
    push!(history.rhos, rho)
    push!(history.ωs, ω)
  end 
end

function update_history!(
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
  push!(history.residuals, square_sum(r))
  push!(history.ωs, ω)
  push!(history.rhos, rho)
  push!(history.αDivs, αDiv)
  push!(history.ωDivs, ωDiv)
  push!(history.leadingCoeffs, leadingCoeff)
end

function observable_failure(
    observablePrecision :: T,
    ω :: T,
    rho :: T,
    αDiv :: T,
    ωDiv :: T,
    leadingCoeff :: T
  ) :: Bool where {T <: Number}

  ω < observablePrecision || rho < observablePrecision || αDiv < observablePrecision || ωDiv < observablePrecision || leadingCoeff < observablePrecision
end

function get_r_shadow(
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

function _bicgstab( 
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
    ϵ :: T = sqrt(eps(T)),
    resetEvery :: ITER = UInt16(1002),
    # perturbated has decent effect for FVM-noprec, random works great, 
    rShadowStrategy :: R_SHADOW_STRATEGY = random,
    observablePrecision :: T = 10e-4 # questionable
  ) where {T <: Number, TID <: Integer, ITER <: Number}


  r = Vector{Float64}(undef, length(xIter))
  r .= b .- mul!(r, A, xIter)
  r_shadow = get_r_shadow(r, b, rShadowStrategy)
  past_rho = α = β = ω = one(T)
  i = history.nIters
  
  t = r # sztuczka - oszczędza wektor
  rho = dot(r_shadow, r)
  init_history!(history, debug, nIter, xIter, r, rho, ω)
  while i <= nIter

    # possible issues with ω and past_rho
    β = (rho / past_rho) * (α / ω) # order?
    # from benchmarking it seems like this is as effective as loop
    p .= r .+ β .* (p .- ω .* v)
    ldiv!(y, K, p)
    mul!(v, A, y) # somehow this is way faster than my attempts to manually calc it
    αDiv = dot(r_shadow, v)
    α = rho / αDiv
    s .= r .- α .* v
    ldiv!(z, K, s)
    mul!(t, A, z)
    # dot is always > 0 but why not observe
    # this can be more effective but somehow manually calcing it is slower
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
    # have fun with resets later
    # reset on certain iterations
    #= if observable_failure(observablePrecision, ω, rho, αDiv, ωDiv, leadingCoeff)
      println("RESET")
      history.status = reset
      history.nResets = history.nResets + one(ITER)
      history.nIters = i
      return
    end =#
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

function bicgstab!(
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
  # undef are faster but all in all this is useless xd
  y = Vector{T}(undef, n)
  s = Vector{T}(undef, n)
  z = Vector{T}(undef, n)

  while is_bicgstab_continuing(history)
    
    if (perturbateX) # what?????????????????
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

