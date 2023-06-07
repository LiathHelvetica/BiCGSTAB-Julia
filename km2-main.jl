# fvmproblem(n :: Integer)
# fdmproblem(n :: Integer)
# femproblem(n :: Integer)
# SparseMatrixCSC{Tv, Ti <: Integer} - drugi generyk odpowiada za przechowanie indeksów

# include - pastes entire file (i wonder if it works for modules)
# import - imports namespace, exports must be used with the namespace
# using - imports every exported name

# τ w ilu
# Dla 0 przeprowadza pełną faktoryzację (niedobrze)
# Domyślnie 0.001
# Następuje drop jeżeli τ * abs(v) <= max z rzędu
# Przynajmniej taka jest ogólna idea - ta biblioteka zdaje się nie mieć dokumentacji

# https://docs.juliahub.com/IterativeSolvers/ef2NV/0.8.5/getting_started/#ConvergenceHistory

# TODO: add plots
# TODO: add matrix from the market

include("MatrixGenerators/nonsymmetric-problem-fvm.jl")
include("MatrixGenerators/symmetric-problem-fdm.jl")
include("MatrixGenerators/symmetric-problem-fem.jl")

import IterativeSolvers: bicgstabl, nprods
import Krylov: bicgstab
import IncompleteLU: ilu
# using Plots
using PyPlot

function norm2(A, b, x)
  norm(b - A * x, 2)
end

function print_data_is(A, b, x, hs, tag)
  println("$(tag): $(norm2(A, b, x)), iterations: $(hs.iters), number of MV: $(nprods(hs)), converged: $(hs.isconverged)")
end

function print_data_k(A, b, x, hs, tag)
  println("$(tag): $(norm2(A, b, x)), iterations: $(hs.niter), converged: $(hs.solved)")
end

MAX_ITER = 250
MAX_MV = 2 * MAX_ITER
τ = 0.001

A_fv, b_fv = fvmproblem(500)
A_fd, b_fd = fdmproblem(25)
A_fe, b_fe = femproblem(17)
A_fv_prec = ilu(A_fv, τ = τ)
A_fd_prec = ilu(A_fd, τ = τ)
A_fe_prec = ilu(A_fe, τ = τ)

x_fv_is_1, hs_fv_is_1 = bicgstabl(A_fv, b_fv, 1, log = true, max_mv_products = MAX_MV)
x_fd_is_1, hs_fd_is_1 = bicgstabl(A_fd, b_fd, 1, log = true, max_mv_products = MAX_MV)
x_fe_is_1, hs_fe_is_1 = bicgstabl(A_fe, b_fe, 1, log = true, max_mv_products = MAX_MV)

println("Solutions for BiCGSTAB(1) from IterativeSolvers - no preconditioning")
print_data_is(A_fv, b_fv, x_fv_is_1, hs_fv_is_1, "FVM")
# hs_fv_is_1[:resnorm] - wektor residuali
# zamiast tego plot(hs_fv_is_1)
print_data_is(A_fd, b_fd, x_fd_is_1, hs_fd_is_1, "FDM")
print_data_is(A_fe, b_fe, x_fe_is_1, hs_fe_is_1, "FEM")

x_fv_is_2, hs_fv_is_2 = bicgstabl(A_fv, b_fv, 2, log = true, max_mv_products = MAX_MV)
x_fd_is_2, hs_fd_is_2 = bicgstabl(A_fd, b_fd, 2, log = true, max_mv_products = MAX_MV)
x_fe_is_2, hs_fe_is_2 = bicgstabl(A_fe, b_fe, 2, log = true, max_mv_products = MAX_MV)

println("Solutions for BiCGSTAB(2) from IterativeSolvers - no preconditioning")
print_data_is(A_fv, b_fv, x_fv_is_2, hs_fv_is_2, "FVM")
print_data_is(A_fd, b_fd, x_fd_is_2, hs_fd_is_2, "FDM")
print_data_is(A_fe, b_fe, x_fe_is_2, hs_fe_is_2, "FEM")

x_fv_k_1, hs_fv_k_1 = bicgstab(A_fv, b_fv, history = true, itmax = MAX_ITER)
x_fd_k_1, hs_fd_k_1 = bicgstab(A_fd, b_fd, history = true, itmax = MAX_ITER)
x_fe_k_1, hs_fe_k_1 = bicgstab(A_fe, b_fe, history = true, itmax = MAX_ITER)

println("Solutions for BiCGSTAB from Krylov - no preconditioning")
print_data_k(A_fv, b_fv, x_fv_k_1, hs_fv_k_1, "FVM")
# hs_fv_k_1.residuals
print_data_k(A_fd, b_fd, x_fd_k_1, hs_fd_k_1, "FDM")
print_data_k(A_fe, b_fe, x_fe_k_1, hs_fe_k_1, "FEM")

x_fv_is_1_p, hs_fv_is_1_p = bicgstabl(A_fv, b_fv, 1, log = true, max_mv_products = MAX_MV, Pl = A_fv_prec)
x_fd_is_1_p, hs_fd_is_1_p = bicgstabl(A_fd, b_fd, 1, log = true, max_mv_products = MAX_MV, Pl = A_fd_prec)
x_fe_is_1_p, hs_fe_is_1_p = bicgstabl(A_fe, b_fe, 1, log = true, max_mv_products = MAX_MV, Pl = A_fe_prec)

println("Solutions for BiCGSTAB(1) from IterativeSolvers - preconditioned")
print_data_is(A_fv, b_fv, x_fv_is_1_p, hs_fv_is_1_p, "FVM")
print_data_is(A_fd, b_fd, x_fd_is_1_p, hs_fd_is_1_p, "FDM")
print_data_is(A_fe, b_fe, x_fe_is_1_p, hs_fe_is_1_p, "FEM")

x_fv_is_2_p, hs_fv_is_2_p = bicgstabl(A_fv, b_fv, 2, log = true, max_mv_products = MAX_MV, Pl = A_fv_prec)
x_fd_is_2_p, hs_fd_is_2_p = bicgstabl(A_fd, b_fd, 2, log = true, max_mv_products = MAX_MV, Pl = A_fd_prec)
x_fe_is_2_p, hs_fe_is_2_p = bicgstabl(A_fe, b_fe, 2, log = true, max_mv_products = MAX_MV, Pl = A_fe_prec)

println("Solutions for BiCGSTAB(2) from IterativeSolvers - preconditioned")
print_data_is(A_fv, b_fv, x_fv_is_2_p, hs_fv_is_2_p, "FVM")
print_data_is(A_fd, b_fd, x_fd_is_2_p, hs_fd_is_2_p, "FDM")
print_data_is(A_fe, b_fe, x_fe_is_2_p, hs_fe_is_2_p, "FEM")

x_fv_k_1_p, hs_fv_k_1_p = bicgstab(A_fv, b_fv, history = true, itmax = MAX_ITER, M = A_fv_prec, ldiv = true)
x_fd_k_1_p, hs_fd_k_1_p = bicgstab(A_fd, b_fd, history = true, itmax = MAX_ITER, M = A_fd_prec, ldiv = true)
x_fe_k_1_p, hs_fe_k_1_p = bicgstab(A_fe, b_fe, history = true, itmax = MAX_ITER, M = A_fe_prec, ldiv = true)

println("Solutions for BiCGSTAB from Krylov - left preconditioning")
print_data_k(A_fv, b_fv, x_fv_k_1_p, hs_fv_k_1_p, "FVM")
print_data_k(A_fd, b_fd, x_fd_k_1_p, hs_fd_k_1_p, "FDM")
print_data_k(A_fe, b_fe, x_fe_k_1_p, hs_fe_k_1_p, "FEM")

x_fv_k_1_p_r, hs_fv_k_1_p_r = bicgstab(A_fv, b_fv, history = true, itmax = MAX_ITER, N = A_fv_prec, ldiv = true)
x_fd_k_1_p_r, hs_fd_k_1_p_r = bicgstab(A_fd, b_fd, history = true, itmax = MAX_ITER, N = A_fd_prec, ldiv = true)
x_fe_k_1_p_r, hs_fe_k_1_p_r = bicgstab(A_fe, b_fe, history = true, itmax = MAX_ITER, N = A_fe_prec, ldiv = true)

println("Solutions for BiCGSTAB from Krylov - left preconditioning")
print_data_k(A_fv, b_fv, x_fv_k_1_p_r, hs_fv_k_1_p_r, "FVM")
print_data_k(A_fd, b_fd, x_fd_k_1_p_r, hs_fd_k_1_p_r, "FDM")
print_data_k(A_fe, b_fe, x_fe_k_1_p_r, hs_fe_k_1_p_r, "FEM")

#plot(
#  hs_fv_is_2[:resnorm],
#  label = "fv",
#  title = "BiCGSTAB(2) - ID - no preconditioning",
#  xlabel = "iterations",
#  ylabel = "residual"
#)
#plot!(
#  hs_fd_is_2[:resnorm],
#  label = "fd",
#)
#plt = plot!(
#  hs_fe_is_2[:resnorm],
#  label = "fe",
#)

# display(plt)

fig, ax = subplots(1)
ax.plot(
  hs_fv_is_2[:resnorm],
  label = "FVM"
)
ax.plot(
  hs_fd_is_2[:resnorm],
  label = "FDM"
)
ax.plot(
  hs_fe_is_2[:resnorm],
  label = "FEM"
)

title("BiCGSTAB(2) - IS - no preconditioning")
xlabel("iteration")
ylabel("residual")
legend()
show()


#=
julia> @benchmark bicgstabl(A_fv, b_fv, 1)

BenchmarkTools.Trial: 1184 samples with 1 evaluation.
 Range (min … max):  3.013 ms … 9.593 ms  ┊ GC (min … max): 0.00% … 27.48%
 Time  (median):     3.911 ms             ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.202 ms ± 1.080 ms  ┊ GC (mean ± σ):  5.32% ± 10.89%

  ▁▇▆█▁  ▁▃  ▁▁                                              
  █████████████▇▇▆▆▆▆▅▄▄▅▅▄▄▄▄▄▅▄▃▄▃▃▄▃▃▂▃▃▂▃▂▃▂▃▁▂▁▂▂▂▃▁▂▂ ▄
  3.01 ms        Histogram: frequency by time       7.91 ms <

 Memory estimate: 4.01 MiB, allocs estimate: 1264.
=#
#=
julia> @benchmark bicgstabl(A_fd, b_fd, 1)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  207.827 μs …   4.143 ms  ┊ GC (min … max): 0.00% … 88.71%
 Time  (median):     252.310 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   310.999 μs ± 217.017 μs  ┊ GC (mean ± σ):  4.68% ±  6.70%

  ▁ █▂▁▁                                                         
  █▇████▄▄▃▃▃▃▃▃▃▃▂▂▂▂▃▃▃▃▃▃▃▃▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  208 μs           Histogram: frequency by time          678 μs <

 Memory estimate: 268.77 KiB, allocs estimate: 84.
=#
#=
julia> @benchmark bicgstabl(A_fe, b_fe, 1)
BenchmarkTools.Trial: 4466 samples with 1 evaluation.
 Range (min … max):  684.026 μs …   5.812 ms  ┊ GC (min … max): 0.00% … 72.32%
 Time  (median):     943.794 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.105 ms ± 458.179 μs  ┊ GC (mean ± σ):  3.93% ±  9.04%

    ▄█▅▂                                                         
  ▃▇████▇▆▆▅▅▅▄▄▅▅▅▄▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂ ▃
  684 μs           Histogram: frequency by time         3.23 ms <

 Memory estimate: 694.27 KiB, allocs estimate: 204.
=#
