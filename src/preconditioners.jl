using LinearAlgebra, SparseArrays
import IncompleteLU: ILUFactorization

export incomplete_cholesky, modified_incomplete_cholesky

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

function modified_incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = spzeros(T, n, n)
    ffj = 0.01
    A_copy = copy(A)

    
    for j = 1:n
        L[j, j] = sqrt(A_copy[j, j])
        w = spzeros(T, n)

        w1_j = spzeros(T, j)
        wj_1_n = A_copy[j+1:end, j]
        w = [w1_j; wj_1_n]
        
        for k = 1:j-1
           
            
            for i = j+1:n

                if L[j, k] != zero(T)
                    w[i] -= L[i, k] * L[j, k]
                end
            end
        end
        
        for i = j+1:n
            w[i] /= L[j, j]
        end
        
        ffj = ffj * norm(w,2)
       
        for i = j+1:n
            if abs(w[i]) < ffj
                w[i] = zero(T)
            end
        end
        
        
        p = Int(floor(maximum(A_copy[:, j])))

        for k = 1:p
            
            L[k, j] -= w[k]          
        end       

            
        for i = j+1:n
            A_copy[i, i] -= L[i,j]^2          
        end


    end

    Lt = sparse((transpose(L)))
    lu_factorization = ILUFactorization(Lt, L)
    return lu_factorization 
end
