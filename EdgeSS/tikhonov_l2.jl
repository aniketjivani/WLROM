using LinearAlgebra

function first_diffs_1d_matrix(n)
    D = diagm(0 => -ones(n), 1 => ones(n - 1), -(n - 1) => [1])
    return D
end

function tikhonov_l2(A, b, lambda)

    D = first_diffs_1d_matrix(size(A, 2))
    Atil = vcat(A, lambda * D)
    btil = vcat(b, zeros(size(D, 1)))

    x = Atil \ btil

    return x
end