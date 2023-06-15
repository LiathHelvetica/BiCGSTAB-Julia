using SuiteSparse
import IncompleteLU: ILUFactorization
function incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = spzeros(T, n, n)
    A_copy = copy(A)

    @inbounds @views for k = 1:n
        L[k, k] = sqrt(A_copy[k, k])

        @inbounds @views for i = k+1:n
            if A_copy[i, k] != zero(T)
                L[i, k] = A_copy[i, k] / L[k, k]
            end
        end

        @inbounds @views for j = k+1:n
            @inbounds   @views for i = j:n
                if A_copy[i, j] != zero(T)
                    A_copy[i, j] -= L[i, k] * L[j, k]
                end
            end
        end
    end

    Lt = sparse((transpose(L)))
    lu_factorization = ILUFactorization(Lt, L)
    return lu_factorization 
end
