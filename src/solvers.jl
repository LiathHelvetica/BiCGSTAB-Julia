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

=#

import LinearAlgebra: dot, I, UniformScaling, norm
import IncompleteLU: ILUFactorization

export bicgstab, bicgstab!

function square_sum(vv :: AbstractVector{T}) :: T where {T <: Number}
  out = zero(T)
  for v in vv
    out = out + v^2
  end
  out
end

function _bicgstab(
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    K :: Union{AbstractMatrix{T}, ILUFactorization{T, TID}, UniformScaling{Bool}} = I,
    ϵ :: T = eps(T),
    nIter :: ITER = UInt16(1000)
  ) :: AbstractVector{T} where {T <: Number, TID <: Integer, ITER <: Number}

  r = b - A * xIter
  r_shadow = r # questionable
  past_rho = α = β = ω = 1
  v = p = 0
  i = zero(ITER)
  rho = dot(r_shadow, r)
  while true

    # this is weird since this method below actually solves FVM without prec.
    # even though it virtually is the same as the other one

    # preallocate all - można odciążyć pamięć poprzez mutowalne operacje np mul!

    β = (rho / past_rho) * (α / ω) # order?
    p = r .+ β * (p - ω * v) # is dot less efficient?
    y = K \ p
    v = A * y
    α = rho / dot(r_shadow, v)
    s = r - α * v # preallocate 
    z = K \ s
    t = A * z # preallocate
    ω = dot(t, s) / dot(t, t)
    past_rho = rho
    rho = -ω * dot(r_shadow, t)
    xIter = xIter + α * y + ω * z
    r = s - ω * t 
    if square_sum(r) < ϵ || i > nIter
      return xIter
    end
    i = i + one(ITER)
  end
end

function bicgstab(
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    kwargs...
  ) :: AbstractVector{T} where {T <: Number}
  bicgstab!(zeros(length(b)), A, b; kwargs...)
end

function bicgstab!(
    xIter :: AbstractVector{T},
    A :: AbstractMatrix{T},
    b :: AbstractVector{T};
    perturbateX :: Bool = true,
    τ :: T = 0.001,
    kwargs...
  ) :: AbstractVector{T} where {T <: Number}
  if (perturbateX)
    perturbation = rand(T, length(xIter))
    xIter = τ * norm(xIter, 2) *  perturbation / norm(perturbation, 2) 
  end
  _bicgstab(xIter, A, b; kwargs...)
end

