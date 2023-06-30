include("MatrixGenerators/nonsymmetric-problem-fvm.jl")
include("MatrixGenerators/symmetric-problem-fdm.jl")
include("MatrixGenerators/symmetric-problem-fem.jl")

include("src/BiCGSTAB.jl")

import IncompleteLU: ilu
import BenchmarkTools: @benchmark, Trial
import BenchmarkTools
import IterativeSolvers: bicgstabl
import Krylov: bicgstab as bicgstab_kr
import DataFrames: DataFrame

using PyPlot
using .BiCGSTAB

# ustawić te samą niepewność
# dodać metryki: err, nIters 

function bench_as_raw_str(t :: Trial)
  io = IOBuffer()
  show(io, "text/plain", t)
  split(String(take!(io)), "\n")
end

function get_mem(data)
  part = split(data[10], ",")[1]
  part = split(part, ":")[2]
  string(strip(part))
end

function get_nallocs(data :: Trial)
  string(data.allocs)
end

function get_time_med(data)
  part = split(data[3], "┊")[1]
  part = split(part, ":")[2]
  string(strip(part))
end

function get_time_mean(data)
  part = split(data[4], "┊")[1]
  part = split(part, ":")[2]
  string(strip(part))
end

# Rows: memory, nallocs, time (mean), time (median), nIters, err, converged
function get_bench_data(t :: Trial, converged :: Bool, err :: Float64, nIters :: Int)
  data = bench_as_raw_str(t)
  [
    get_mem(data),
    get_nallocs(t),
    get_time_mean(data),
    get_time_med(data),
    string(nIters),
    string(err),
    converged ? "yes" : "no"
  ]
end

function norm2(A, b, x)
  norm(b - A * x, 2)
end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

FVM_SEED = zeros(2000)
FVM_SEED[200:400] .= 1.0
MAX_ITER = 350
MAX_MV = 2 * MAX_ITER
τ = 0.001

A_fv, b_fv = fvmproblem(FVM_SEED) # 2000 x 2000
A_fd, b_fd = fdmproblem(47) # 2025 x 2025
A_fe, b_fe = femproblem(33) # 2112 x 2112

Ilu_fv = ilu(A_fv, τ = τ)
Ilu_fd = ilu(A_fd, τ = τ)
Ilu_fe = ilu(A_fe, τ = τ)

Ch_fv = incomplete_cholesky(A_fv)
Ch_fd = incomplete_cholesky(A_fd)
Ch_fe = incomplete_cholesky(A_fe)

Ch_mod_fv = modified_incomplete_cholesky(A_fv)
Ch_mod_fd = modified_incomplete_cholesky(A_fd)
Ch_mod_fe = modified_incomplete_cholesky(A_fe)

x_fv, hs_fv = bicgstab(A_fv, b_fv, nIter = MAX_ITER, debug = true)
x_fd, hs_fd = bicgstab(A_fd, b_fd, nIter = MAX_ITER, debug = true)
x_fe, hs_fe = bicgstab(A_fe, b_fe, nIter = MAX_ITER, debug = true)

x_fv_is, hs_fv_is = bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, log = true)
x_fd_is, hs_fd_is = bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, log = true)
x_fe_is, hs_fe_is = bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, log = true)

x_fv_kr, hs_fv_kr = bicgstab_kr(A_fv, b_fv, history = true, itmax = MAX_ITER)
x_fd_kr, hs_fd_kr = bicgstab_kr(A_fd, b_fd, history = true, itmax = MAX_ITER)
x_fe_kr, hs_fe_kr = bicgstab_kr(A_fe, b_fe, history = true, itmax = MAX_ITER)

x_fv_per, hs_fv_per = bicgstab(A_fv, b_fv, rShadowStrategy = perturbated, nIter = MAX_ITER, debug = true)
x_fd_per, hs_fd_per = bicgstab(A_fd, b_fd, rShadowStrategy = perturbated, nIter = MAX_ITER, debug = true)
x_fe_per, hs_fe_per = bicgstab(A_fe, b_fe, rShadowStrategy = perturbated, nIter = MAX_ITER, debug = true)

x_fv_rand, hs_fv_rand = bicgstab(A_fv, b_fv, rShadowStrategy = random, nIter = MAX_ITER, debug = true)
x_fd_rand, hs_fd_rand = bicgstab(A_fd, b_fd, rShadowStrategy = random, nIter = MAX_ITER, debug = true)
x_fe_rand, hs_fe_rand = bicgstab(A_fe, b_fe, rShadowStrategy = random, nIter = MAX_ITER, debug = true)

x_fv_enr, hs_fv_enr = bicgstab(A_fv, b_fv, rShadowStrategy = enriched, nIter = MAX_ITER, debug = true)
x_fd_enr, hs_fd_enr = bicgstab(A_fd, b_fd, rShadowStrategy = enriched, nIter = MAX_ITER, debug = true)
x_fe_enr, hs_fe_enr = bicgstab(A_fe, b_fe, rShadowStrategy = enriched, nIter = MAX_ITER, debug = true)

x_fv_p, hs_fv_p = bicgstab(A_fv, b_fv, K = Ilu_fv, nIter = MAX_ITER, debug = true)
x_fd_p, hs_fd_p = bicgstab(A_fd, b_fd, K = Ilu_fd, nIter = MAX_ITER, debug = true)
x_fe_p, hs_fe_p = bicgstab(A_fe, b_fe, K = Ilu_fe, nIter = MAX_ITER, debug = true)

x_fv_is_p, hs_fv_is_p = bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ilu_fv, log = true)
x_fd_is_p, hs_fd_is_p = bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ilu_fd, log = true)
x_fe_is_p, hs_fe_is_p = bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ilu_fe, log = true)

x_fv_kr_p, hs_fv_kr_p = bicgstab_kr(A_fv, b_fv, history = true, itmax = MAX_ITER, N = Ilu_fv, ldiv = true)
x_fd_kr_p, hs_fd_kr_p = bicgstab_kr(A_fd, b_fd, history = true, itmax = MAX_ITER, N = Ilu_fd, ldiv = true)
x_fe_kr_p, hs_fe_kr_p = bicgstab_kr(A_fe, b_fe, history = true, itmax = MAX_ITER, N = Ilu_fe, ldiv = true)

x_fv_ch, hs_fv_ch = bicgstab(A_fv, b_fv, K = Ch_fv, nIter = MAX_ITER, debug = true)
x_fd_ch, hs_fd_ch = bicgstab(A_fd, b_fd, K = Ch_fd, nIter = MAX_ITER, debug = true)
x_fe_ch, hs_fe_ch = bicgstab(A_fe, b_fe, K = Ch_fe, nIter = MAX_ITER, debug = true)

x_fv_is_ch, hs_fv_is_ch = bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ch_fv, log = true)
x_fd_is_ch, hs_fd_is_ch = bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ch_fd, log = true)
x_fe_is_ch, hs_fe_is_ch = bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ch_fe, log = true)

x_fv_kr_ch, hs_fv_kr_ch = bicgstab_kr(A_fv, b_fv, history = true, itmax = MAX_ITER, N = Ch_fv, ldiv = true)
x_fd_kr_ch, hs_fd_kr_ch = bicgstab_kr(A_fd, b_fd, history = true, itmax = MAX_ITER, N = Ch_fd, ldiv = true)
x_fe_kr_ch, hs_fe_kr_ch = bicgstab_kr(A_fe, b_fe, history = true, itmax = MAX_ITER, N = Ch_fe, ldiv = true)

x_fv_ch_mod, hs_fv_ch_mod = bicgstab(A_fv, b_fv, K = Ch_mod_fv, nIter = MAX_ITER, debug = true)
x_fd_ch_mod, hs_fd_ch_mod = bicgstab(A_fd, b_fd, K = Ch_mod_fd, nIter = MAX_ITER, debug = true)
x_fe_ch_mod, hs_fe_ch_mod = bicgstab(A_fe, b_fe, K = Ch_mod_fe, nIter = MAX_ITER, debug = true)

x_fv_is_ch_mod, hs_fv_is_ch_mod = bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fv, log = true)
x_fd_is_ch_mod, hs_fd_is_ch_mod = bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fd, log = true)
x_fe_is_ch_mod, hs_fe_is_ch_mod = bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fe, log = true)

x_fv_kr_ch_mod, hs_fv_kr_ch_mod = bicgstab_kr(A_fv, b_fv, history = true, itmax = MAX_ITER, N = Ch_mod_fv, ldiv = true)
x_fd_kr_ch_mod, hs_fd_kr_ch_mod = bicgstab_kr(A_fd, b_fd, history = true, itmax = MAX_ITER, N = Ch_mod_fd, ldiv = true)
x_fe_kr_ch_mod, hs_fe_kr_ch_mod = bicgstab_kr(A_fe, b_fe, history = true, itmax = MAX_ITER, N = Ch_mod_fe, ldiv = true)


our_bench_fv = @benchmark bicgstab(A_fv, b_fv, nIter = MAX_ITER)
our_bench_fd = @benchmark bicgstab(A_fd, b_fd, nIter = MAX_ITER)
our_bench_fe = @benchmark bicgstab(A_fe, b_fe, nIter = MAX_ITER)

is_bench_fv = @benchmark bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV)
is_bench_fd = @benchmark bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV)
is_bench_fe = @benchmark bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV)

kr_bench_fv = @benchmark bicgstab_kr(A_fv, b_fv, itmax = MAX_ITER)
kr_bench_fd = @benchmark bicgstab_kr(A_fd, b_fd, itmax = MAX_ITER)
kr_bench_fe = @benchmark bicgstab_kr(A_fe, b_fe, itmax = MAX_ITER)

our_bench_fv_per = @benchmark bicgstab(A_fv, b_fv, rShadowStrategy = perturbated, nIter = MAX_ITER)
our_bench_fd_per = @benchmark bicgstab(A_fd, b_fd, rShadowStrategy = perturbated, nIter = MAX_ITER)
our_bench_fe_per = @benchmark bicgstab(A_fe, b_fe, rShadowStrategy = perturbated, nIter = MAX_ITER)

our_bench_fv_rand = @benchmark bicgstab(A_fv, b_fv, rShadowStrategy = random, nIter = MAX_ITER)
our_bench_fd_rand = @benchmark bicgstab(A_fd, b_fd, rShadowStrategy = random, nIter = MAX_ITER)
our_bench_fe_rand = @benchmark bicgstab(A_fe, b_fe, rShadowStrategy = random, nIter = MAX_ITER)

our_bench_fv_enr = @benchmark bicgstab(A_fv, b_fv, rShadowStrategy = enriched, nIter = MAX_ITER)
our_bench_fd_enr = @benchmark bicgstab(A_fd, b_fd, rShadowStrategy = enriched, nIter = MAX_ITER)
our_bench_fe_enr = @benchmark bicgstab(A_fe, b_fe, rShadowStrategy = enriched, nIter = MAX_ITER)

our_bench_fv_ilu = @benchmark bicgstab(A_fv, b_fv, K = Ilu_fv, nIter = MAX_ITER)
our_bench_fd_ilu = @benchmark bicgstab(A_fd, b_fd, K = Ilu_fd, nIter = MAX_ITER)
our_bench_fe_ilu = @benchmark bicgstab(A_fe, b_fe, K = Ilu_fe, nIter = MAX_ITER)

is_bench_fv_ilu = @benchmark bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ilu_fv)
is_bench_fd_ilu = @benchmark bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ilu_fd)
is_bench_fe_ilu = @benchmark bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ilu_fe)

kr_bench_fv_ilu = @benchmark bicgstab_kr(A_fv, b_fv, itmax = MAX_ITER, N = Ilu_fv, ldiv = true)
kr_bench_fd_ilu = @benchmark bicgstab_kr(A_fd, b_fd, itmax = MAX_ITER, N = Ilu_fd, ldiv = true)
kr_bench_fe_ilu = @benchmark bicgstab_kr(A_fe, b_fe, itmax = MAX_ITER, N = Ilu_fe, ldiv = true)

our_bench_fv_ch = @benchmark bicgstab(A_fv, b_fv, K = Ch_fv, nIter = MAX_ITER)
our_bench_fd_ch = @benchmark bicgstab(A_fd, b_fd, K = Ch_fd, nIter = MAX_ITER)
our_bench_fe_ch = @benchmark bicgstab(A_fe, b_fe, K = Ch_fe, nIter = MAX_ITER)

is_bench_fv_ch = @benchmark bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ch_fv)
is_bench_fd_ch = @benchmark bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ch_fd)
is_bench_fe_ch = @benchmark bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ch_fe)

kr_bench_fv_ch = @benchmark bicgstab_kr(A_fv, b_fv, itmax = MAX_ITER, N = Ch_fv, ldiv = true)
kr_bench_fd_ch = @benchmark bicgstab_kr(A_fd, b_fd, itmax = MAX_ITER, N = Ch_fd, ldiv = true)
kr_bench_fe_ch = @benchmark bicgstab_kr(A_fe, b_fe, itmax = MAX_ITER, N = Ch_fe, ldiv = true)

our_bench_fv_ch_mod = @benchmark bicgstab(A_fv, b_fv, K = Ch_mod_fv, nIter = MAX_ITER)
our_bench_fd_ch_mod = @benchmark bicgstab(A_fd, b_fd, K = Ch_mod_fd, nIter = MAX_ITER)
our_bench_fe_ch_mod = @benchmark bicgstab(A_fe, b_fe, K = Ch_mod_fe, nIter = MAX_ITER)

is_bench_fv_ch_mod = @benchmark bicgstabl(A_fv, b_fv, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fv)
is_bench_fd_ch_mod = @benchmark bicgstabl(A_fd, b_fd, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fd)
is_bench_fe_ch_mod = @benchmark bicgstabl(A_fe, b_fe, 1, max_mv_products = MAX_MV, Pl = Ch_mod_fe)

kr_bench_fv_ch_mod = @benchmark bicgstab_kr(A_fv, b_fv, itmax = MAX_ITER, N = Ch_mod_fv, ldiv = true)
kr_bench_fd_ch_mod = @benchmark bicgstab_kr(A_fd, b_fd, itmax = MAX_ITER, N = Ch_mod_fd, ldiv = true)
kr_bench_fe_ch_mod = @benchmark bicgstab_kr(A_fe, b_fe, itmax = MAX_ITER, N = Ch_mod_fe, ldiv = true)

# Rows: memory, nallocs, time (mean), time (median), nIters, err, converged

# FV - standard
fv_df = DataFrame() # err, nIters
fv_df[:, :Our] = get_bench_data(our_bench_fv, hs_fv.status == converged, norm2(A_fv, b_fv, x_fv), hs_fv.nIters)
fv_df[:, :IS] = get_bench_data(is_bench_fv, hs_fv_is.isconverged, norm2(A_fv, b_fv, x_fv_is), hs_fv_is.iters)
fv_df[:, :Krylov] = get_bench_data(kr_bench_fv, hs_fv_kr.solved, norm2(A_fv, b_fv, x_fv_kr), hs_fv_kr.niter)

println(fv_df)

# FD - standard
fd_df = DataFrame()
fd_df[:, :Our] = get_bench_data(our_bench_fd, hs_fd.status == converged, norm2(A_fd, b_fd, x_fd), hs_fd.nIters)
fd_df[:, :IS] = get_bench_data(is_bench_fd, hs_fd_is.isconverged, norm2(A_fd, b_fd, x_fd_is), hs_fd_is.iters)
fd_df[:, :Krylov] = get_bench_data(kr_bench_fd, hs_fd_kr.solved, norm2(A_fd, b_fd, x_fd_kr), hs_fd_kr.niter)

println(fd_df)

# FE - standard
fe_df = DataFrame()
fe_df[:, :Our] = get_bench_data(our_bench_fe, hs_fe.status == converged, norm2(A_fe, b_fe, x_fe), hs_fe.nIters)
fe_df[:, :IS] = get_bench_data(is_bench_fe, hs_fe_is.isconverged, norm2(A_fe, b_fe, x_fe_is), hs_fe_is.iters)
fe_df[:, :Krylov] = get_bench_data(kr_bench_fe, hs_fe_kr.solved, norm2(A_fe, b_fe, x_fe_kr), hs_fe_kr.niter)

println(fe_df)

# FV - different r_shadow
fv_df_rs = DataFrame()
fv_df_rs[:, :initial] = get_bench_data(our_bench_fv, hs_fv.status == converged, norm2(A_fv, b_fv, x_fv), hs_fv.nIters)
fv_df_rs[:, :perturbated] = get_bench_data(our_bench_fv_per, hs_fv_per.status == converged, norm2(A_fv, b_fv, x_fv_per), hs_fv_per.nIters)
fv_df_rs[:, :random] = get_bench_data(our_bench_fv_rand, hs_fv_rand.status == converged, norm2(A_fv, b_fv, x_fv_rand), hs_fv_rand.nIters)
fv_df_rs[:, :enriched] = get_bench_data(our_bench_fv_enr, hs_fv_enr.status == converged, norm2(A_fv, b_fv, x_fv_enr), hs_fv_enr.nIters)

println(fv_df_rs)

# FD - different r_shadow
fd_df_rs = DataFrame()
fd_df_rs[:, :initial] = get_bench_data(our_bench_fd, hs_fd.status == converged, norm2(A_fd, b_fd, x_fd), hs_fd.nIters)
fd_df_rs[:, :perturbated] = get_bench_data(our_bench_fd_per, hs_fd_per.status == converged, norm2(A_fd, b_fd, x_fd_per), hs_fd_per.nIters)
fd_df_rs[:, :random] = get_bench_data(our_bench_fd_rand, hs_fd_rand.status == converged, norm2(A_fd, b_fd, x_fd_per), hs_fd_per.nIters)
fd_df_rs[:, :enriched] = get_bench_data(our_bench_fd_enr, hs_fd_enr.status == converged, norm2(A_fd, b_fd, x_fd_per), hs_fd_per.nIters)

println(fd_df_rs)

# FE - different r_shadow
fe_df_rs = DataFrame()
fe_df_rs[:, :initial] = get_bench_data(our_bench_fe, hs_fe.status == converged, norm2(A_fe, b_fe, x_fe), hs_fe.nIters)
fe_df_rs[:, :perturbated] = get_bench_data(our_bench_fe_per, hs_fe_per.status == converged, norm2(A_fe, b_fe, x_fe_per), hs_fe_per.nIters)
fe_df_rs[:, :random] = get_bench_data(our_bench_fe_rand, hs_fe_rand.status == converged, norm2(A_fe, b_fe, x_fe_per), hs_fe_per.nIters)
fe_df_rs[:, :enriched] = get_bench_data(our_bench_fe_enr, hs_fe_enr.status == converged, norm2(A_fe, b_fe, x_fe_per), hs_fe_per.nIters)

println(fe_df_rs)

# FV - ILU
fv_df_ilu = DataFrame()
fv_df_ilu[:, :Our] = get_bench_data(our_bench_fv_ilu, hs_fv_p.status == converged, norm2(A_fv, b_fv, x_fv_p), hs_fv_p.nIters)
fv_df_ilu[:, :IS] = get_bench_data(is_bench_fv_ilu, hs_fv_is_p.isconverged, norm2(A_fv, b_fv, x_fv_is_p), hs_fv_is_p.iters)
fv_df_ilu[:, :Krylov] = get_bench_data(kr_bench_fv_ilu, hs_fv_kr_p.solved, norm2(A_fv, b_fv, x_fv_kr_p), hs_fv_kr_p.niter)

println(fv_df_ilu)

# FD - ILU
fd_df_ilu = DataFrame()
fd_df_ilu[:, :Our] = get_bench_data(our_bench_fd_ilu, hs_fd_p.status == converged, norm2(A_fd, b_fd, x_fd_p), hs_fd_p.nIters)
fd_df_ilu[:, :IS] = get_bench_data(is_bench_fd_ilu, hs_fd_is_p.isconverged, norm2(A_fd, b_fd, x_fd_is_p), hs_fd_is_p.iters)
fd_df_ilu[:, :Krylov] = get_bench_data(kr_bench_fd_ilu, hs_fd_kr_p.solved, norm2(A_fd, b_fd, x_fd_kr_p), hs_fd_kr_p.niter)

println(fd_df_ilu)

# FE - ILU
fe_df_ilu = DataFrame()
fe_df_ilu[:, :Our] = get_bench_data(our_bench_fe_ilu, hs_fe_p.status == converged, norm2(A_fe, b_fe, x_fe_p), hs_fe_p.nIters)
fe_df_ilu[:, :IS] = get_bench_data(is_bench_fe_ilu, hs_fe_is_p.isconverged, norm2(A_fe, b_fe, x_fe_is_p), hs_fe_is_p.iters)
fe_df_ilu[:, :Krylov] = get_bench_data(kr_bench_fe_ilu, hs_fe_kr_p.solved, norm2(A_fe, b_fe, x_fe_kr_p), hs_fe_kr_p.niter)

println(fe_df_ilu)

# FV - Cholesky
fv_df_ch = DataFrame()
fv_df_ch[:, :Our] = get_bench_data(our_bench_fv_ch, hs_fv_ch.status == converged, norm2(A_fv, b_fv, x_fv_ch), hs_fv_ch.nIters)
fv_df_ch[:, :IS] = get_bench_data(is_bench_fv_ch, hs_fv_is_ch.isconverged, norm2(A_fv, b_fv, x_fv_is_ch), hs_fv_is_ch.iters)
fv_df_ch[:, :Krylov] = get_bench_data(kr_bench_fv_ch, hs_fv_kr_ch.solved, norm2(A_fv, b_fv, x_fv_kr_ch), hs_fv_kr_ch.niter)

println(fv_df_ch)

# FD - Cholesky
fd_df_ch = DataFrame()
fd_df_ch[:, :Our] = get_bench_data(our_bench_fd_ch, hs_fd_ch.status == converged, norm2(A_fd, b_fd, x_fd_ch), hs_fd_ch.nIters)
fd_df_ch[:, :IS] = get_bench_data(is_bench_fd_ch, hs_fd_is_ch.isconverged, norm2(A_fd, b_fd, x_fd_is_ch), hs_fd_is_ch.iters)
fd_df_ch[:, :Krylov] = get_bench_data(kr_bench_fd_ch, hs_fd_kr_ch.solved, norm2(A_fd, b_fd, x_fd_kr_ch), hs_fd_kr_ch.niter)

println(fd_df_ch)

# FE - Cholesky
fe_df_ch = DataFrame()
fe_df_ch[:, :Our] = get_bench_data(our_bench_fe_ch, hs_fe_ch.status == converged, norm2(A_fe, b_fe, x_fe_ch), hs_fe_ch.nIters)
fe_df_ch[:, :IS] = get_bench_data(is_bench_fe_ch, hs_fe_is_ch.isconverged, norm2(A_fe, b_fe, x_fe_is_ch), hs_fe_is_ch.iters)
fe_df_ch[:, :Krylov] = get_bench_data(kr_bench_fe_ch, hs_fe_kr_ch.solved, norm2(A_fe, b_fe, x_fe_kr_ch), hs_fe_kr_ch.niter)

println(fe_df_ch)

# FV - Modified Cholesky
fv_df_ch_mod = DataFrame()
fv_df_ch_mod[:, :Our] = get_bench_data(our_bench_fv_ch_mod, hs_fv_ch_mod.status == converged, norm2(A_fv, b_fv, x_fv_ch_mod), hs_fv_ch_mod.nIters)
fv_df_ch_mod[:, :IS] = get_bench_data(is_bench_fv_ch_mod, hs_fv_is_ch_mod.isconverged, norm2(A_fv, b_fv, x_fv_is_ch_mod), hs_fv_is_ch_mod.iters)
fv_df_ch_mod[:, :Krylov] = get_bench_data(kr_bench_fv_ch_mod, hs_fv_kr_ch_mod.solved, norm2(A_fv, b_fv, x_fv_kr_ch_mod), hs_fv_kr_ch_mod.niter)

println(fv_df_ch_mod)

# FD - Modified Cholesky
fd_df_ch_mod = DataFrame()
fd_df_ch_mod[:, :Our] = get_bench_data(our_bench_fd_ch_mod, hs_fd_ch_mod.status == converged, norm2(A_fd, b_fd, x_fd_ch_mod), hs_fd_ch_mod.nIters)
fd_df_ch_mod[:, :IS] = get_bench_data(is_bench_fd_ch_mod, hs_fd_is_ch_mod.isconverged, norm2(A_fd, b_fd, x_fd_is_ch_mod), hs_fd_is_ch_mod.iters)
fd_df_ch_mod[:, :Krylov] = get_bench_data(kr_bench_fd_ch_mod, hs_fd_kr_ch_mod.solved, norm2(A_fd, b_fd, x_fd_kr_ch_mod), hs_fd_kr_ch_mod.niter)

println(fd_df_ch_mod)

# FE - Modified Cholesky
fe_df_ch_mod = DataFrame()
fe_df_ch_mod[:, :Our] = get_bench_data(our_bench_fe_ch_mod, hs_fe_ch_mod.status == converged, norm2(A_fe, b_fe, x_fe_ch_mod), hs_fe_ch_mod.nIters)
fe_df_ch_mod[:, :IS] = get_bench_data(is_bench_fe_ch_mod, hs_fe_is_ch_mod.isconverged, norm2(A_fe, b_fe, x_fe_is_ch_mod), hs_fe_is_ch_mod.iters)
fe_df_ch_mod[:, :Krylov] = get_bench_data(kr_bench_fe_ch_mod, hs_fe_kr_ch_mod.solved, norm2(A_fe, b_fe, x_fe_kr_ch_mod), hs_fe_kr_ch_mod.niter)

println(fe_df_ch_mod)

plt.plot(hs_fv.residuals, label="no preconditioning")
plt.plot(hs_fv_rand.residuals, label="random r-shadow")
plt.plot(hs_fv_p.residuals, label="ILU")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.title("Residuals of different solutions of FVM")
plt.legend()
plt.grid()
plt.show()

plt.plot(hs_fd.residuals, label="no preconditioning")
plt.plot(hs_fd_rand.residuals, label="random r-shadow")
plt.plot(hs_fd_p.residuals, label="ILU")
plt.plot(hs_fd_ch.residuals, label="Cholesky")
plt.plot(hs_fd_ch_mod.residuals, label="Mod. Cholesky")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.title("Residuals of different solutions of FDM")
plt.legend()
plt.grid()
plt.show()

plt.plot(hs_fe.residuals, label="no preconditioning")
plt.plot(hs_fe_rand.residuals, label="random r-shadow")
plt.plot(hs_fe_p.residuals, label="ILU")
plt.plot(hs_fe_ch.residuals, label="Cholesky")
plt.plot(hs_fe_ch_mod.residuals, label="Mod. Cholesky")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.title("Residuals of different solutions of FEM")
plt.legend()
plt.grid()
plt.show()
