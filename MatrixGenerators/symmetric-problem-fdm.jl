using SparseArrays

function fdmproblem(n)
    m = n
    A = zeros(n*m, n*m)
    dof = reshape(1:n*m, n, m)
    for i=2:n, j=1:m
        A[dof[i,j], dof[i-1,j]] -= 1.0
        A[dof[i,j], dof[i,j]] += 1.0
    end
    for i=1:n, j=2:m
        A[dof[i,j], dof[i,j-1]] -= 1.0
        A[dof[i,j], dof[i,j]] += 1.0
    end
    for i=1:n-1, j=1:m
        A[dof[i,j], dof[i+1,j]] -= 1.0
        A[dof[i,j], dof[i,j]] += 1.0
    end
    for i=1:n, j=1:m-1
        A[dof[i,j], dof[i,j+1]] -= 1.0
        A[dof[i,j], dof[i,j]] += 1.0
    end
    for i=(1,n), j=1:m
        A[dof[i,j], :] .= 0.0
        A[dof[i,j], dof[i,j]] = 1.0
    end
    for i=1:n, j=(1,m)
        A[dof[i,j], :] .= 0.0
        A[:, dof[i,j]] .= 0.0
        A[dof[i,j], dof[i,j]] = 1.0
    end    
    x = zeros(n*m)
    for i=1:n, j=1:m
        x[dof[i,j]] += sin(1.0π * (i-1)/(n-1))
        x[dof[i,j]] += sin(1.0π * (j-1)/(m-1))
    end
    # A is of size (n^2, n^2)
    b = A * x
    # free is a vector of length (n-2)^2
    free = vec([dof[i, j] for i=2:n-1, j=2:m-1])
    # resulting A is matrix of size (n-2)^2 
    return sparse(A[free, free]), b[free]
end
