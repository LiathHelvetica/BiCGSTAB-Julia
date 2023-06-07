using LinearAlgebra
using SparseArrays

stiffness_element(lx, ly) =
[lx/ly  1  lx/ly  1
   1  ly/lx  1  ly/lx
 lx/ly  1  lx/ly  1
   1  ly/lx  1  ly/lx]

mass_element(lx, ly) = (lx * ly) / 6.0 *
[ 2. 0 -1  0
  0  2  0 -1
 -1  0  2  0
  0 -1  0  2]

function assemble!(S, R, T, el2ed, el2edd, dof, lx, ly, nelem, ndof; εr=1, μr=1, ε0=8.854e-12, μ0=4e-7π)
  # ASSUMPTION: Waveguide is homogenous
  # Assemble stiffness and mass matrices
  ε = εr*ε0
  for ielem = 1:nelem # Assemble by elements
    Se = stiffness_element(lx, ly)
    Te = mass_element(lx, ly)
    
    for jedge = 1:4
      dj = el2edd[ielem, jedge]
      jj = dof[el2ed[ielem, jedge]]
      if jj == 0
        continue
      end
      
      for kedge = 1:4
        dk = el2edd[ielem, kedge]
        kk = dof[el2ed[ielem, kedge]]
        if kk == 0
          continue
        end
  
        S[jj, kk] = S[jj, kk] + dj * dk * (1/μr) * Se[jedge, kedge]
        T[jj, kk] = T[jj, kk] + dj * dk * (μ0*ε) * Te[jedge, kedge]
      end
    end
  end
  return nothing
end

function lhs(S, T, R, Δt)
  A = (+0.25Δt^2 * S +  T + 0.5Δt * R)
end

function rhs(S, T, R, Δt, ep, epp)
  b = (-0.25Δt^2 * S -  T + 0.5Δt * R) * epp +
      (-0.50Δt^2 * S + 2T) * ep
end

function quadmesh(a, b, Nx, Ny)
    NUM_EDGES = 2(Nx*Ny) + Nx + Ny
    NUM_ELEMS = Nx * Ny
    
    el2edd = repeat([+1 +1 -1 -1], NUM_ELEMS)
    el2ed = zeros(Int64, NUM_ELEMS, 4)
    for jj = 1:Ny
        for ii = 1:Nx
            kk = (jj-1)Nx + ii
            el2ed[kk, :]  .= [ii, ii+Nx+1, ii+Nx+1+Nx, ii+Nx]
            el2ed[kk, :] .+= (jj-1) * (Nx + Nx + 1)
        end
    end

    return el2ed, el2edd, NUM_EDGES
end

function femproblem(n)
    m = n
    @assert n * m < 40*40 "Are you sure? Try smaller problem first (n < 40)!"
    
    # parameters
    Δt = 0.01e-9
    Lx = 2.00
    Ly = 2.00
    lx = Lx / n
    ly = Ly / m
    el2ed, el2edd, nedge = quadmesh(Lx, Ly, n, m);
    
    # degrees of freedom
    DOF_NONE = 0
    DOF_PEC  = 1

    h = [  1+(2n+1)i: n+0+(2n+1)i for i=0:m]
    v = [n+1+(2n+1)i:2n+1+(2n+1)i for i=0:m-1]

    Γ = zeros(Int64, nedge)
    Γ[first(h)] .= DOF_PEC
    Γ[last(h)]  .= DOF_PEC
    for i=1:m
         Γ[first(v[i])] = DOF_PEC
         Γ[last(v[i])] = DOF_PEC
    end

    dof = collect(1:nedge)
    free = Γ .!= DOF_PEC

    # assemble finite element matrices
    S = zeros(nedge, nedge)
    T = zeros(nedge, nedge)
    R = zeros(nedge, nedge)
    assemble!(S, T, R, el2ed, el2edd, dof, lx, ly, n*m, nedge)
    
    # construct the problem left hand side
    A = lhs(S[free, free], T[free, free], R[free, free], Δt);
    
    # calculate eigensolution and use it as a starting point
    k², v = eigen(Array(S[free, free]), Array(T[free, free]))
    
    e = zeros(nedge)
    ep = copy(e)
    epp = copy(e)
    ep[free] .= epp[free] .= v[:, 1+(n-1)*(m-1)]
    
    # construct the problem right hand side
    b = rhs(S[free, free], T[free, free], R[free, free], Δt, ep[free], epp[free])
    
    return 1e23sparse(A), 1e23b
end
