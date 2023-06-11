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

Czy istnieje sposó” aby łatwo podejrzeć ciało metody

=#

import LinearAlgebra: dot, I, UniformScaling, norm, ldiv!, mul!
import IncompleteLU: ILUFactorization

export bicgstab, bicgstab!

struct History{T <: Number, ITER <: Number}
  xs :: Vector{AbstractVector{T}}
  residuals :: Vector{T}
  nIters :: ITER
end

function square_sum(vv :: AbstractVector{T}) :: T where {T <: Number}
  out = zero(T)
  for v in vv
    out = out + v^2
  end 
  out
end

function init_history(
    debug :: Bool,
    nIter :: ITER,
    xIter :: AbstractVector{T},
    r :: AbstractVector{T}
  ) :: History{T, ITER} where {T <: Number, ITER <: Number}

  if debug
    # undef? is it effective?
    out = History{T, ITER}(
      Vector{Vector{T}}(undef, nIter + one(nIter)), 
      Vector{T}(undef, nIter + one(nIter)),
      zero(ITER)
    )
    out.xs[1] = xIter
    out.residuals[1] = square_sum(r)
    return out
  end 
  
  History{T, ITER}(Vector{Vector{T}}(), Vector{T}(), zero(ITER))
end

function update_history(
    history :: History{T, ITER},
    xIter :: AbstractVector{T},
    r :: AbstractVector{T},
    i :: ITER
  ) where {T <: Number, ITER <: Number}

  id = i + one(i)
  history.xs[id] = copy(xIter)
  history.residuals[id] = square_sum(r)
end

function tidy_history(
    history :: History{T, ITER},
    nIters :: ITER
  ) where {T <: Number, ITER <: Number}

  nVals = nIters + one(nIters)
  # views ? this is whatever
  History(history.xs[1 : nVals], history.residuals[1 : nVals], nIters)
end

function _bicgstab(
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    K :: Union{AbstractMatrix{T}, ILUFactorization{T, TID}, UniformScaling{Bool}} = I,
    ϵ :: T = sqrt(eps(T)),
    nIter :: ITER = UInt16(1000),
    debug :: Bool = false
  ) where {T <: Number, TID <: Integer, ITER <: Number}

  n = length(xIter)
  r = b - A * xIter
  history = init_history(debug, nIter, xIter, r)
  r_shadow = copy(r) # questionable - should be different value
  past_rho = α = β = ω = one(T)
  i = one(ITER)
  v = zeros(T, n)
  p = zeros(T, n)

  # undef are faster but all in all this is useless xd
  y = Vector{T}(undef, n)
  s = Vector{T}(undef, n)
  z = Vector{T}(undef, n)
  t = r # sztuczka - oszczędza wektor
  rho = dot(r_shadow, r)
  while true

    # possible issues with ω and past_rho
    β = (rho / past_rho) * (α / ω) # order?
    # from benchmarking it seems like this is as effective as loop
    p .= r .+ β .* (p .- ω .* v)
    ldiv!(y, K, p)
    mul!(v, A, y) # somehow this is way faster than my attempts to manually calc it
    α = rho / dot(r_shadow, v)
    s .= r .- α .* v
    ldiv!(z, K, s)
    mul!(t, A, z)
    # dot is always > 0 but why not observe
    # this can be more effective but somehow manually calcing it is slower
    ω = dot(t, s) / dot(t, t)
    past_rho = rho
    rho = -ω * dot(r_shadow, t)
    xIter .= xIter .+ α .* y .+ ω .* z
    r .= s .- ω .* t 
    if debug
      update_history(history, xIter, r, i)
    end
    if square_sum(r) < ϵ || i >= nIter
      if debug
        return (xIter, tidy_history(history, i))
      else 
        return xIter
      end
    end
    i = i + one(ITER)
  end
end

function bicgstab(
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    kwargs...
  ) where {T <: Number}

  bicgstab!(zeros(length(b)), A, b; kwargs...)
end

function bicgstab!(
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    perturbateX :: Bool = true,
    τ :: T = 0.001,
    kwargs...
  ) where {T <: Number}

  if (perturbateX)
    perturbation = rand(T, length(xIter))
    xIter .= τ .* norm(xIter, 2) .*  perturbation / norm(perturbation, 2) 
  end
  _bicgstab(xIter, A, b; kwargs...)
end

