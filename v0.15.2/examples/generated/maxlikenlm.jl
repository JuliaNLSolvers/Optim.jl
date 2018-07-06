using Optim, NLSolversBase
srand(0);                            # Fix random seed generator for reproducibility

n = 500                             # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [ones(n) randn(n, nvar - 1)]    # X matrix of explanatory variables plus constant
ε = randn(n) * 0.5                  # Error variance
y = x * β + ε;                      # Generate Data

function Log_Likelihood(X, Y, β, log_σ)
    σ = exp(log_σ)
    llike = -n/2*log(2π) - n/2* log(σ^2) - (sum((Y - X * β).^2) / (2σ^2))
    llike = -llike
end

func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar + 1]),
                           ones(nvar+1); autodiff=:forward);

opt = optimize(func, ones(nvar+1))

parameters = Optim.minimizer(opt)

parameters[nvar+1] = exp(parameters[nvar+1])

numerical_hessian = hessian!(func,parameters)

var_cov_matrix = inv(numerical_hessian)

β = parameters[1:nvar]

temp = diag(var_cov_matrix)
temp1 = temp[1:nvar]

t_stats = β./sqrt.(temp1)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

