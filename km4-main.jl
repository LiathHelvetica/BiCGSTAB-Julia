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

Ch_fv = incomplete_cholesky(A_fv)
Ch_fd = incomplete_cholesky(A_fd)
Ch_fe = incomplete_cholesky(A_fe)

Ch_mod_fv = modified_incomplete_cholesky(A_fv)
Ch_mod_fd = modified_incomplete_cholesky(A_fd)
Ch_mod_fe = modified_incomplete_cholesky(A_fe)

x_fv = bicgstab(A_fv, b_fv)
x_fd = bicgstab(A_fd, b_fd)
x_fe = bicgstab(A_fe, b_fe)

x_fv_per = bicgstab(A_fv, b_fv, rShadowStrategy = perturbated)
x_fd_per = bicgstab(A_fd, b_fd, rShadowStrategy = perturbated)
x_fe_per = bicgstab(A_fe, b_fe, rShadowStrategy = perturbated)

x_fv_rand = bicgstab(A_fv, b_fv, rShadowStrategy = random)
x_fd_rand = bicgstab(A_fd, b_fd, rShadowStrategy = random)
x_fe_rand = bicgstab(A_fe, b_fe, rShadowStrategy = random)

x_fv_enr = bicgstab(A_fv, b_fv, rShadowStrategy = enriched)
x_fd_enr = bicgstab(A_fd, b_fd, rShadowStrategy = enriched)
x_fe_enr = bicgstab(A_fe, b_fe, rShadowStrategy = enriched)

x_fv_p = bicgstab(A_fv, b_fv, K = K_fv)
x_fd_p = bicgstab(A_fd, b_fd, K = K_fd)
x_fe_p = bicgstab(A_fe, b_fe, K = K_fe)

x_fv_ch = bicgstab(A_fv, b_fv, K = Ch_fv)
x_fd_ch = bicgstab(A_fd, b_fd, K = Ch_fd)
x_fe_ch = bicgstab(A_fe, b_fe, K = Ch_fe)

x_fv_ch_mod = bicgstab(A_fv, b_fv, K = Ch_mod_fv)
x_fd_ch_mod = bicgstab(A_fd, b_fd, K = Ch_mod_fd)
x_fe_ch_mod = bicgstab(A_fe, b_fe, K = Ch_mod_fe)

println("##### no preconditioning #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv))
println("FDM")
println(norm2(A_fd, b_fd, x_fd))
println("FEM")
println(norm2(A_fe, b_fe, x_fe))
println("##### ilu #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv_p))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_p))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_p))
println("##### Cholesky #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv_ch))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_ch))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_ch))
println("##### Modified Cholesky #####")
# println("FVM")
# println(norm2(A_fv, b_fv, x_fv_ch_mod))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_ch_mod))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_ch_mod))
println("##### no prec - perturbated shadow #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv_per))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_per))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_per))
println("##### no prec - random shadow #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv_rand))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_rand))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_rand))
println("##### no prec - enriched shadow #####")
println("FVM")
println(norm2(A_fv, b_fv, x_fv_enr))
println("FDM")
println(norm2(A_fd, b_fd, x_fd_enr))
println("FEM")
println(norm2(A_fe, b_fe, x_fe_enr))


(x_fv_dbg, hs_fv_dbg) = bicgstab(A_fv, b_fv, debug = true)
(x_fd_dbg, hs_fd_dbg) = bicgstab(A_fd, b_fd, debug = true)
(x_fe_dbg, hs_fe_dbg) = bicgstab(A_fe, b_fe, debug = true)

(x_fv_p_dbg, hs_fv_p_dbg) = bicgstab(A_fv, b_fv, K = K_fv, debug = true)
(x_fd_p_dbg, hs_fd_p_dbg) = bicgstab(A_fd, b_fd, K = K_fd, debug = true)
(x_fe_p_dbg, hs_fe_p_dbg) = bicgstab(A_fe, b_fe, K = K_fe, debug = true)

