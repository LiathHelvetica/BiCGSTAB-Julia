import IncompleteLU: ILUFactorization
import SparseArrays: SparseMatrixCSC, sparse

export incomplete_cholesky

# nnz - liczba niezerowych elementów
# findnz - do postaci transportowej - raczej nie będzie efektywne
# rowvals - wektor - dla kolejnych wartości jaki rząd ich?
# nzrange(A, i) - range - range wartości z tablicy wartości - gdzie się one znajdują dla kolumny i
# nonzeros - wektor wartości niezerowych
#
# W sumie i tak aby zainicjalizować macierz to trzeba podać ją w postaci 3 wektorów
#
# A.colptr - w którym indeksie zaczyna się kolumna
# A.rowval - tej samej długości co vec elementów - w jakim rzędzie się znajdują
#
function incomplete_cholesky(A :: SparseMatrixCSC{T, TID}) :: ILUFactorization{T, TID} where {T <: Number, TID <: Integer}
  out_row_ids :: Vector{TID} = zeros(TID, nnz(A))
  out_col_ids :: Vector{TID} = zeros(TID, nnz(A))
  out_vals :: Vector{T} = zeros(T, nnz(A))
  c = 1
  i = 1
  col_pointers = A.colptr
  row_pointers = A.rowval
  vals = A.nzval

  # might be good to preallocate Vectors - even if overkill

  while c < length(col_pointers)
    for val_id in range(col_pointers[c], col_pointers[c + 1] - 1)
      # column - c
      # row - row_pointers[val_id]
      if c <= row_pointers[val_id]
        out_row_ids[i] = row_pointers[val_id]
        out_col_ids[i] = c
        out_vals[i] = vals[val_id]
        i = i + 1
      end
    end
    c = c + 1
  end

  i = i - 1
  out = sparse(
         out_row_ids[1 : i],
         out_col_ids[1 : i],
         out_vals[1 : i],
         A.m,
         A.n
        )

  ILUFactorization(out, out)

  #=
  c = 1
  while c < length(col_pointers) - 1
    column_sum = zero(T)
    diag_ele = one(T)
    for val_id in range(col_pointers[c], col_pointers[c + 1])
      # column - c
      # row - row_pointers[val_id]
      if row_pointers[val_id] < c
        column_sum = column_sum + vals[val_id] / diag_ele
      elseif row_pointers[val_id] == c
        diag_ele = vals[val_id]
      end
    end
    for val_id in range(col_pointers[c], col_pointers[c + 1])
      if row_pointers[val_id] < c
        out = vals[val_id] / diag_ele
        
      end
    end
    c = c + 1
  end
  =#
end
