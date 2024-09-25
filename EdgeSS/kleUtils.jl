using LinearAlgebra
using Combinatorics
using Statistics

include("./admm.jl")
include("./tikhonov_l2.jl")

function getCenteredSolution(Y)
    if size(Y, 2) == 0
	YMean = zeros(size(Y, 1))
	YCentered = zeros(size(Y, 1))
    else
	YTransformed = Y
	YMean = mean(YTransformed, dims = 2)[:]
	YCentered = YTransformed .- YMean
    end
    return YCentered, YMean
end

function getWeights(grid)
    nGridPts = length(grid)
    wts = zeros(nGridPts)
    # timeGrid = 1:nGridPts
    wts[1] = 0.5 * (grid[2] - grid[1])
    for i = 2:nGridPts - 1
	    wts[i] = 0.5 * (grid[i + 1] - grid[i - 1])
    end
    wts[end] = 0.5 * (grid[end] - grid[end - 1])
    wtRoot = sqrt.(wts)
    wtInvRoot = 1 ./ wtRoot
    return wts, wtRoot, wtInvRoot
end

function getWeightedAreas(A)
    m,n = size(A)
    W = zeros(m + 1, n + 1)
    
    # Left Boundary 
    W[1,1] = A[1,1]
    W[2:m, 1] = A[1:m-1,1] + A[2:m, 1]
    W[m + 1,1] = A[m,1]
    
    
    # Interior
    for j = 2:n
        # Edge
        W[1, j] = A[1, j - 1] + A[1, j]
        W[2:m, j] = A[1:m-1, j-1] + A[1:m-1, j] + A[2:m, j-1] + A[2:m, j]
        W[m + 1, j] = A[m, j - 1] + A[m, j]
    end
    
    # Right Boundary
    W[1,n + 1] = A[1,n]
    W[2:m, n + 1] = A[1:m-1,n] + A[2:m, n]
    W[m + 1, n + 1] = A[m, n]
    
    W = 0.25*0.5*(W + W')
    
    return W
end

function getWeights2D(xgrid)
    dx = diff(xgrid)
    dy = diff(xgrid)

    area = dx*dy'

    W = getWeightedAreas(area)
    wts = W[:]
    wtRoot = sqrt.(wts)
    wtInvRoot = 1 ./ wtRoot

    return wts, wtRoot, wtInvRoot
end


function getSortedEigenpairsFromSVD(YCentered, grid; 
				    useFullGrid = 1,
				    getAllModes = 0,
				    weightFunction = getWeights,
				    )
    ng, N = size(YCentered)
    # wts, wtRoot, wtInvRoot = weightFunction(ng)
    wts, wtRoot, wtInvRoot = weightFunction(grid)
    
    # Form weighted solution matrix B
    B = (1 / sqrt(N - 1)) * (wtRoot .* YCentered)

    F = svd(B)
    λAll = (F.S).^2
    QAll = wtInvRoot .* F.U
    
    if getAllModes == 0
	kt = findall(cumsum(λAll) ./ sum(λAll) .>= 0.95)[1]
	λ  = λAll[1:kt]
	Q  = QAll[:, 1:kt]
	return Q, λ, kt
    else
	kt = length(λAll)
	return QAll, λAll, kt
    end
end


function getUncorrelatedRV(gridWeights, YCentered, Q, λ)

    ζ = (gridWeights .* YCentered)' * Q ./ (sqrt.(λ))'
    return ζ
end

function generatePCEIndex(;order=3, dims=2)
    multiIndex = []
    pMax = deepcopy(order)
    d 	 = deepcopy(dims)
    for p = 0:pMax
        cc = collect(combinations(1:p+d-1, d-1))
        c = permutedims(hcat(cc...))
        m = size(c, 1)
        t = ones(Int, m, p + d - 1)
        t[repeat(1:m, inner=(1, d - 1)) + m * (c .- 1)] .= 0
        u = [zeros(Int, 1, m); t' ; zeros(Int, 1, m)]
        v = cumsum(u, dims = 1)
        x = diff(reshape(v[u .== 0], d + 1, m), dims = 1)'
        push!(multiIndex, x)
    end
    return vcat(multiIndex...)[2:end, :]
end	

function RecursivePolynomial(ξ; order=3, dims=2, family="Legendre")
    pMax = deepcopy(order)
    nPoints, nDims = size(ξ)
    results = []
    if family == "Legendre"
        for d = 1:nDims
            push!(results, zeros(nPoints, pMax + 1))
            for p = 0:pMax
                if p == 0
                results[d][:, p + 1] = ones(nPoints, 1)
                elseif p == 1
                results[d][:, p + 1] = ξ[:, d]
                else
                results[d][:, p + 1] = ((2 * (p - 1) + 1) * ξ[:, d] .* 
                            results[d][:, p] - (p - 1) * results[d][:, p - 1]) / p
                end
            end

            # Get orthonormal basis
            for p = 0:pMax
                results[d][:, p + 1] = results[d][:, p + 1] / sqrt(1 / (2 * p + 1)) # shouldn't this be sqrt(2/(2p + 1))?
            end
        end
        return results
    elseif family == "Hermite"
        for d = 1:nDims
            push!(results, zeros(nPoints, pMax + 1))
            for p = 0:pMax
                if p == 0
                results[d][:, p + 1] = ones(nPoints, 1)
                elseif p == 1
                results[d][:, p + 1] = ξ[:, d]
                else
                results[d][:, p + 1] = ξ[:, d] .* results[d][:, p] - (p - 1) * results[d][:, p - 1]
                end
            end

            # Get orthonormal basis
            for p = 0:pMax
                results[d][:, p + 1] = results[d][:, p + 1] / sqrt(factorial(p))
            end
        end
        return results
    end
end

function PrepCaseA(ξ; order=3, dims=2, family="Legendre")
    multiIndex = generatePCEIndex(;order=order, dims=dims)
    m = size(ξ, 1)
    n = size(multiIndex, 1)
    orthPolynomial = RecursivePolynomial(ξ; order=order, dims=dims, family=family)
    d = deepcopy(dims)
    A = zeros(m, n)

    for i = 1:n
	temp = ones(Int, m)
	for j = 1:d
	    temp = temp .* orthPolynomial[j][:, multiIndex[i, j] + 1]
	end
	A[:, i] = temp
    end

    A = [ones(Int, m) A]

    return A
end


function getPCCoefficients(ξ, ζ; order=3, dims=2, family="Legendre", solver="Tikhonov-L2", lambdas = 10 .^(-2:0.2:3), nFolds=5)
    # we will build PC Coefficients using Legendre Polynomials
    A = PrepCaseA(ξ; order=order, dims=dims, family=family)
    # B = deepcopy(ζ)

    nt = size(A, 2)
    kt = size(ζ, 2)

    bβ = zeros(kt, nt)

    lambda_opt_cv = []
    for k = 1:kt
        err_CV = zeros(length(lambdas))
        
        for it_lam = 1:length(lambdas)
            err_CV[it_lam] = pceCV(A, ζ[:, k], order=order, dims=dims, lambda=lambdas[it_lam], nFolds=nFolds, solver=solver)
        end
        push!(lambda_opt_cv, argmin(err_CV))
        bβ[k, :] = subsolve(A, ζ[:, k], lambda_opt_cv[k]; solver=solver)
    end
    
    return bβ, lambda_opt_cv
end

"""
Function that takes in germ ξ and solution Y and returns KLE decomposition of the same,
i.e. Q(x), λ(x) and ζ(ξ)
"""
function buildKLE(ξ, Y, grid; order=3, dims=2, family="Legendre", useFullGrid=1, getAllModes=0, weightFunction=getWeights, solver="Tikhonov-L2")
    YCentered, YMean = getCenteredSolution(Y)
    wts, wtRoot, wtInvRoot = weightFunction(grid)
    Q, λ, kt = getSortedEigenpairsFromSVD(
                                  YCentered, grid; 
                                  useFullGrid=useFullGrid,
                                  getAllModes=getAllModes,
                                  weightFunction=weightFunction
                                  )
    ζ = getUncorrelatedRV(wts, YCentered, Q, λ)

    bβ, lambdaCV = getPCCoefficients(ξ, ζ; order=order, dims=dims, family=family, solver=solver, lambdas = 10 .^(-2:0.2:3), nFolds=5)




    # bβ = pceFunction(ξ, ζ; order=order, dims=dims, family=family)
    # if doCV
    #     bβ, reg = pceCV(ξ, ζ; order=order, dims=dims, family=family, pceFunction=pceFunction, solver=solver)
    #     return Q, λ, bβ, YMean, reg
    # else
    #     return Q, λ, bβ, YMean
    # end

    return Q, λ, bβ, lambdaCV, YMean
end

"""
Function to use learned KLE coeffs and predict field for new samples of ξ
"""
function predictKLE(ξTest, Q, λ, bβ, YMean; order=3, dims=2)
    YPred = zeros(length(YMean), size(ξTest, 1))
    for testIdx in 1:size(ξTest, 1)
        ΨTest = PrepCaseA(ξTest[testIdx, :]'; order=order, dims=dims)'
        klModes = Q .* sqrt.(λ)'
        YPred[:, testIdx] = klModes * bβ * ΨTest + YMean
    end
    return YPred
end


"""
Function to solve PCE when a regularization parameter is supplied along with data
"""
function getPCCoefficientsReg(ξ, ζ; family="Legendre", order=3, dims=2, λ = 0.5)
    A = PrepCaseA(ξ; order=order, dims=dims, family=family)
    B = deepcopy(ζ)

    nt = size(A, 2)
    kt = size(ζ, 2)

    bβ = zeros(kt, nt)
	
    for k = 1:kt
        bβ[k, :] = (A' * A + λ * I) \ (A' * B[:, k])
    end
    
    return bβ
end

"""
Perform K-fold CV for regression problem on PCE coefficients and return the coefficients, as well as the optimal regularization parameter value based on grid search in log space.
"""
function pceCV(A, y; order=order, dims=dims, lambda=0.01, nFolds=5, solver="Tikhonov-L2")

    rng = MersenneTwister(2022)

    # A = PrepCaseA(ξ; order=order, dims=dims, family=family)
    kFoldErrors = []

    
    m = size(A, 1)
    lo_size = floor(Int, m / nFolds)

    shuffledIdx = shuffle(rng, 1:m)
    AShuffled = A[shuffledIdx, :]
    YShuffled = y[shuffledIdx]

    err_CV_fold = zeros(nFolds)
    lo_range = 0
    for lo = 1:nFolds
        if lo < nFolds
            lo_range = (lo_range[end] + 1):(lo_range[end] + lo_size)
        elseif lo==nFolds
            lo_range = (lo_range[end] + 1):m
        end
        
        A_CV = AShuffled[setdiff(1:m, lo_range), :]
        A_test = AShuffled[lo_range, :]
        y_CV = YShuffled[setdiff(1:m, lo_range)]

        b_CV = subsolve(A_CV, y_CV, lambda; solver=solver)
        y_pred = A_test * b_CV
        err_CV_fold[lo] = norm(y_pred - YShuffled[lo_range])^2

    end
    kFoldErrors = sqrt(sum(err_CV_fold)) / norm(y)
    return kFoldErrors
end


function subsolve(A, b, lambda; solver="Tikhonov-L2")
    if solver=="Tikhonov-L2"
        x = tikhonov_l2(A, b, lambda)
    elseif solver=="ADMM"
        x = admm(A, b, 0.5 * lambda, 1.0, 1.0; quiet=true)
    else
        error("Solver $(solver) is not available, try one of Tikhonov-L2 or ADMM")
    end

    return x
end