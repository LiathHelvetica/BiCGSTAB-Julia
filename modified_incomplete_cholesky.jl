using LinearAlgebra, SparseArrays

function modified_incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = spzeros(T, n, n)
    ffj = 0.001
    A_copy = copy(A)

    
    @inbounds for j = 1:n
        L[j, j] = sqrt(A_copy[j, j])
        w = spzeros(T, n)

        w1_j = spzeros(T, j)
        wj_1_n = A_copy[j+1:end, j]
        w =  [w1_j; wj_1_n]
        
        @inbounds for k = 1:j-1
           
            
            for i = j+1:n

                if L[j, k] != zero(T)
                    w[i] -= L[i, k] * L[j, k]
                end
            end
        end
        
        @inbounds for i = j+1:n
            w[i] /=  L[j, j]
        end
        
        ffj = ffj * norm(w,2)
       
        @inbounds for i = j+1:n
            if abs(w[i]) < ffj
                w[i] = zero(T)
            end
        end
        
        
        p = Int(floor(maximum(A_copy[:, j])))

        @inbounds for k = 1:p
            
            L[k, j] -=  w[k]          
        end       

            
        @inbounds for i = j+1:n
            A_copy[i, i] -= L[i,j]^2          
        end


    end

    Lt = sparse((transpose(L)))
    lu_factorization = ILUFactorization(Lt, L)
    return lu_factorization 
end
