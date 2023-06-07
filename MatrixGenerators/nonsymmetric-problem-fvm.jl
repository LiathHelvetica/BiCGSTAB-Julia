using LinearAlgebra
using SparseArrays

fvmproblem(n::Integer) = fvmproblem(ones(n))
    
function fvmproblem(f)
    u = 0.2
    ω = 0.5
    θ = 0.5
    dx = 1e-3
    dt = 1e-3
    n = length(f)
    
    A = diagm(-1 => (1-ω) * ones(n-1), +1 => (1-ω) * ones(n-1), 0 => 2ω * ones(n))
    B = diagm(-1 => -u/dx * ones(n-1), +1 => +u/dx * ones(n-1))

    lhs = sparse(A + dt * θ * B)
    rhs =       (A - dt * (1-θ) * B) * f
    
    return lhs, rhs
end

# f  = zeros(1000)
# f[200:300] .= 1.
# A, b = fvmproblem(f)
