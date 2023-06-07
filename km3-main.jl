include("MatrixGenerators/nonsymmetric-problem-fvm.jl")
include("MatrixGenerators/symmetric-problem-fdm.jl")
include("MatrixGenerators/symmetric-problem-fem.jl")

include("src/BiCGSTAB.jl")

import IncompleteLU: ilu

using .BiCGSTAB


function norm2(A, b, x)
  norm(b - A * x, 2)
end

MAX_ITER = 250
τ = 0.001

A_fv, b_fv = fvmproblem(500)
A_fd, b_fd = fdmproblem(25)
A_fe, b_fe = femproblem(17)

K_fv = ilu(A_fv, τ = τ)
K_fd = ilu(A_fd, τ = τ)
K_fe = ilu(A_fe, τ = τ)

x_fv = bicgstab(A_fv, b_fv)
x_fd = bicgstab(A_fd, b_fd)
x_fe = bicgstab(A_fe, b_fe)

x_fv_p = bicgstab(A_fv, b_fv, K = K_fv)
x_fd_p = bicgstab(A_fd, b_fd, K = K_fd)
x_fe_p = bicgstab(A_fe, b_fe, K = K_fe)

println("FVM - no preconditioning")
println(norm2(A_fv, b_fv, x_fv))
println("FDM - no preconditioning")
println(norm2(A_fd, b_fd, x_fd))
println("FEM - no preconditioning")
println(norm2(A_fe, b_fe, x_fe))
println("FVM - preconditioned")
println(norm2(A_fv, b_fv, x_fv_p))
println("FDM - preconditioned")
println(norm2(A_fd, b_fd, x_fd_p))
println("FEM - preconditioned")
println(norm2(A_fe, b_fe, x_fe_p))
