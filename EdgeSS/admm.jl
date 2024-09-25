using Printf
using LinearAlgebra

"""
lasso  Solve lasso problem via ADMM
    `[z, history] = lasso(A, b, lambda, rho, alpha);`
Solves the following problem via ADMM:
  minimize 1/2*|| Ax - b ||_2^2 + λ || x ||_1
Solution is returned in the vector z - TYPO in original doc - don't confuse with intermediate variable x in the code!!!!
history is a structure that contains the objective value, the primal and
dual residual norms, and the tolerances for the primal and dual residual
norms at each iteration.
rho is the augmented Lagrangian parameter.
alpha is the over-relaxation parameter (typical values for alpha are between 1.0 and 1.8).
More information can be found in the paper linked at: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
"""
function admm(A, b, lambda, rho, alpha; z0=nothing, quiet=false, max_iter=1000, abstol=1e-4, reltol=1e-2)
    # Global constants and defaults
    
    # Data preprocessing
    m, n = size(A);
    
    # save a matrix-vector multiply
    Atb = A'*b;
    
    # ADMM solver
    x = zeros(n);
    
    # add warm start for z
    if !(isnothing(z0))
        z = z0;
    else
        z = zeros(n);
    end
    u = zeros(n);

    # cache the factorization
    L, U = factor(A, rho);

    if !(quiet)
        @printf("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n", "iter", "r norm", "eps pri", "s norm", "eps dual", "objective")
    end

    # initialize history object
    history = admmHistory(max_iter)
    
    # Start loop for ADMM
    for k = 1:max_iter
        # x-update
        q = Atb + rho*(z - u);    # temporary value
        if m >= n    # if skinny
            x = U \ (L \ q);
        else            # if fat
            x = (q/rho) - (A'*(U \ ( L \ (A*q) )))/rho^2;
        end
        
        # z-update with relaxation
        zold = z;
        x_hat = alpha*x + (1 - alpha)*zold;
        z = shrinkage(x_hat + u, lambda/rho);

        # u-update
        u = u + (x_hat - z);

        # diagnostics, reporting, termination checks
        history.objval[k]  = objective(A, b, lambda, x, z);

        history.r_norm[k]  = norm(x - z);
        history.s_norm[k]  = norm(-rho*(z - zold));

        history.eps_pri[k] = sqrt(n)*abstol + reltol*max(norm(x), norm(-z));
        history.eps_dual[k]= sqrt(n)*abstol + reltol*norm(rho*u);

        if !(quiet)
            @printf("%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n", k,
                    history.r_norm[k], history.eps_pri[k],
                    history.s_norm[k], history.eps_dual[k], history.objval[k]);
        end

        if (history.r_norm[k] < history.eps_pri[k]) &&
           (history.s_norm[k] < history.eps_dual[k])
            break;
        end
    end
    return z, history
end

function objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) )
    return p
end

function shrinkage(x, kappa)
    z = max.( 0, x .- kappa ) - max.( 0, -x .- kappa );
    return z
end

function factor(A, rho)
    m, n = size(A);
    if m >= n        # if skinny
        C = cholesky(A'*A + rho*I)
        L = C.L
        U = C.U
       # L = chol(A'*A + rho*speye(n), 'lower');
    else             # if fat
        C = cholesky((1/rho)*(A*A') + I)
        L = C.L
        U = C.U
       # L = chol(speye(m) + 1/rho*(A*A'), 'lower');
    end
    return L, U
end

"""
Type containing fields that log objective function value and other intermediate values during the ADMM LASSO update.
"""        
mutable struct admmHistory
    objval::Vector
    r_norm::Vector
    s_norm::Vector
    eps_pri::Vector
    eps_dual::Vector
    max_iter
    function admmHistory(max_iter)
        return new(zeros(max_iter), zeros(max_iter), zeros(max_iter), zeros(max_iter), zeros(max_iter))
    end
end
        
                
