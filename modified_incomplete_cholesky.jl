using LinearAlgebra, SparseArrays

function modified_incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = spzeros(T, n, n)
    ffj = 0.01
    println(n)
    
    for j = 1:n
        L[j, j] = sqrt(A[j, j])
        w = zeros(T, n)

        w1_j = zeros(T, j)
        wj_1_n = A[j+1:end, j]
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
 
        ffj = ffj * norm(w)
       # print(ffj)

        for i = j+1:n
            if abs(w[i]) < ffj
                w[i] = zero(T)
            end
        end

            
        for i = j+1:n
            A[i, i] -= L[i,j]^2          
        end
# muszę dokończyć z p

    end
    
    return ilu(L)
end
