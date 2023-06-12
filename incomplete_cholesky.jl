using SuiteSparse
import IncompleteLU: ILUFactorization
function incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = zeros(T, n, n)
    
    for k = 1:n
        L[k, k] = √(A[k, k])
        
        for i = k+1:n
            if A[i, k] != zero(T)
                L[i, k] = A[i, k] / L[k, k]
            end
        end
        
        for j = k+1:n
            for i = j:n
                if A[i, j] != zero(T)
                    A[i, j] -= L[i, k] * L[j, k]
                end
            end
        end
    end
    
    for i = 1:n
        for j = i+1:n
            L[i, j] = zero(T)
        end
    end
    
    L_sparse = sparse(L)
    lu_factorization = ILUFactorization(A, L_sparse)
    return lu_factorization

end
