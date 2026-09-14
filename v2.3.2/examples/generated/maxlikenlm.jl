using Optim, NLSolversBase
using LinearAlgebra: diag
using ForwardDiff
using ADTypes: AutoForwardDiff

n = 40                              # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [
    1.0 0.156651# X matrix of explanatory variables plus constant
    1.0 -1.34218
    1.0 0.238262
    1.0 -0.496572
    1.0 1.19352
    1.0 0.300229
    1.0 0.409127
    1.0 -0.88967
    1.0 -0.326052
    1.0 -1.74367
    1.0 -0.528113
    1.0 1.42612
    1.0 -1.08846
    1.0 -0.00972169
    1.0 -0.85543
    1.0 1.0301
    1.0 1.67595
    1.0 -0.152156
    1.0 0.26666
    1.0 -0.668618
    1.0 -0.36883
    1.0 -0.301392
    1.0 0.0667779
    1.0 -0.508801
    1.0 -0.352346
    1.0 0.288688
    1.0 -0.240577
    1.0 -0.997697
    1.0 -0.362264
    1.0 0.999308
    1.0 -1.28574
    1.0 -1.91253
    1.0 0.825156
    1.0 -0.136191
    1.0 1.79925
    1.0 -1.10438
    1.0 0.108481
    1.0 0.847916
    1.0 0.594971
    1.0 0.427909
]

ε = [
    0.5539830489065279             # Errors
    -0.7981494315544392
    0.12994853889935182
    0.23315434715658184
    -0.1959788033050691
    -0.644463980478783
    -0.04055657880388486
    -0.33313251280917094
    -0.315407370840677
    0.32273952815870866
    0.56790436131181
    0.4189982390480762
    -0.0399623088796998
    -0.2900421677961449
    -0.21938513655749814
    -0.2521429229103657
    0.0006247891825243118
    -0.694977951759846
    -0.24108791530910414
    0.1919989647431539
    0.15632862280544485
    -0.16928298502504732
    0.08912288359190582
    0.0037707641031662006
    -0.016111044809837466
    0.01852191562589722
    -0.762541135294584
    -0.7204431774719634
    -0.04394527523005201
    -0.11956323865320413
    -0.6713329013627437
    -0.2339928433338628
    -0.6200532213195297
    -0.6192380993792371
    0.08834918731846135
    -0.5099307915921438
    0.41527207925609494
    -0.7130133329859893
    -0.531213372742777
    -0.09029672309221337
]

y = x * β + ε;                      # Generate Data

function Log_Likelihood(X, Y, β, log_σ)
    σ = exp(log_σ)
    llike = -n / 2 * log(2π) - n / 2 * log(σ^2) - (sum((Y - X * β) .^ 2) / (2σ^2))
    llike = -llike
end

func = TwiceDifferentiable(
    vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar+1]),
    ones(nvar + 1);
    autodiff = AutoForwardDiff(),
);

opt = optimize(func, ones(nvar + 1))

parameters = Optim.minimizer(opt)

numerical_hessian = hessian!(func, parameters)

function transform(parameters)
    parameters[end] = exp(parameters[end])
    parameters
end

J = ForwardDiff.jacobian(transform, parameters)'
parameters = transform(parameters)

var_cov_matrix = J * inv(numerical_hessian) * J'

t_stats = parameters ./ sqrt.(diag(var_cov_matrix))

println("parameter estimates:", parameters)
println("t-statsitics: ", t_stats)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
