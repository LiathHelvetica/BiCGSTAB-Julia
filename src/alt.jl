function _bicgstab(
    A :: AbstractMatrix{T},
    b :: AbstractVector{T},
    preconditioned :: Bool;
    xIter :: AbstractVector{T} = zeros(length(b)),
    K :: Union{AbstractMatrix{T}, ILUFactorization{T, TID}} = Matrix{T}(undef, 0, 0),
    ϵ :: T = eps(T),
    nIter :: ITER = UInt16(1000)
  ) :: AbstractVector{T} where {T <: Number, TID <: Integer, ITER <: Number}

  r = b - A * xIter
  r_shadow = r # questionable
  rho = α = β = ω = 1
  v = p = 0
  i = zero(ITER)
  while true

    # this is weird since this method below actually solves FVM without prec.
    # even though it virtually is the same as the other one

    past_rho = rho
    rho = dot(r_shadow, r)
    β = (rho / past_rho) * (α / ω) # order?
    p = r .+ β * (p - ω * v)     # is dot less effective?
    # different steps depending on if needs to be preconditioned
    v = preconditioned ? begin
      y = K \ p
      A * y
    end : A * p
    α = rho / dot(r_shadow, v)
    s = r - α * v # preallocate
    # different steps depending on if needs to be preconditioned
    # preallocate t
    t = preconditioned ? begin 
      z = K \ s
      A * z
    end : A * s
    ω = dot(t, s) / dot(t, t)
    
    xIter = xIter + α * p + ω * s
    r = s - ω * t   # changed order, indirect checking
    # warunki bez równości
    if square_sum(r) < ϵ || i > nIter
      # println(i)
      return xIter
    end
    i = i + one(ITER)
  end
end
