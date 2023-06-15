using LinearAlgebra, SparseArrays
import IncompleteLU: ILUFactorization

export incomplete_cholesky, modified_incomplete_cholesky

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

function modified_incomplete_cholesky(A::SparseMatrixCSC{T, TID})::ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
    n = size(A, 1)
    L = spzeros(T, n, n)
    ffj = 0.01
    sqrt_Ajj = sqrt(A[1, 1])  # LICZYMY POZA PĘTLĄ


    w = similar(L, n)  # PRELOKACJA POZA PĘTLĄ

    for j = 1:n
        L[j, j] = sqrt_Ajj
        
        @views w[1:j] .= 0
        @views w[j+1:end] .= A[j+1:end, j]
        
        for k = 1:j-1
            if L[j, k] == zero(T)
                continue
            end

            @views w[j+1:n] .-= L[j, k] * L[k, j+1:n]
        end

        for i = j+1:n
            w[i] /= L[j, j]
        end

        ffj *= norm(w, 2)

        for i = j+1:n
            if abs(w[i]) < ffj
                w[i] = zero(T)
            end
        end

        p = Int(floor(maximum(A[:, j])))

      #  stosujemy żeby zaoszczędzić pamięć ale zwiększamy czas - zależy czego oczekujemy
      #  @inbounds for k = 1:p
        for k = 1:p
            L[k, j] -= w[k]
        end

       # @inbounds for i = j+1:n
        for i = j+1:n
            A[i, i] -= L[i, j]^2
        end
    end

    return ILUFactorization(A, L)

end
