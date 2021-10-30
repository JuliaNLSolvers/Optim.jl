# # Maximum Likelihood Estimation: The Normal Linear Model
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`maxlikenlm.ipynb`](@__NBVIEWER_ROOT_URL__examples/generated/maxlikenlm.ipynb)
#-
#
# The following tutorial will introduce maximum likelihood estimation
# in Julia for the normal linear model.
#
# The normal linear model (sometimes referred to as the OLS model) is
# the workhorse of regression modeling and is utilized across a number
# of diverse fields.  In this tutorial, we will utilize simulated data
# to demonstrate how Julia can be used to recover the parameters of
# interest.
#
# The first order of business is to use the `Optim` package
# and also include the `NLSolversBase` routine:
#

using Optim, NLSolversBase
using LinearAlgebra: diag
using ForwardDiff

#md # !!! tip
#md #     Add Optim with the following command at the Julia command prompt:
#md #     `Pkg.add("Optim")`
#
# The first item that needs to be addressed is the data generating process or DGP.
# The following code will produce data from a normal linear model:


n = 40                              # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [ 1.0   0.156651				# X matrix of explanatory variables plus constant
 1.0  -1.34218
 1.0   0.238262
 1.0  -0.496572
 1.0   1.19352
 1.0   0.300229
 1.0   0.409127
 1.0  -0.88967
 1.0  -0.326052
 1.0  -1.74367
 1.0  -0.528113
 1.0   1.42612
 1.0  -1.08846
 1.0  -0.00972169
 1.0  -0.85543
 1.0   1.0301
 1.0   1.67595
 1.0  -0.152156
 1.0   0.26666
 1.0  -0.668618
 1.0  -0.36883
 1.0  -0.301392
 1.0   0.0667779
 1.0  -0.508801
 1.0  -0.352346
 1.0   0.288688
 1.0  -0.240577
 1.0  -0.997697
 1.0  -0.362264
 1.0   0.999308
 1.0  -1.28574
 1.0  -1.91253
 1.0   0.825156
 1.0  -0.136191
 1.0   1.79925
 1.0  -1.10438
 1.0   0.108481
 1.0   0.847916
 1.0   0.594971
 1.0   0.427909]

ε = [0.5539830489065279             # Errors
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
 -0.09029672309221337]

y = x * β + ε;                      # Generate Data

# In the above example, we have 500 observations, 2 explanatory
# variables plus an intercept, an error variance equal to 0.5,
# coefficients equal to 3.0, and all of these are subject to change by
# the user. Since we know the true value of these parameters, we
# should obtain these values when we maximize the likelihood function.
#
# The next step in our tutorial is to define a Julia function for the
# likelihood function. The following function defines the likelihood
# function for the normal linear model:

function Log_Likelihood(X, Y, β, log_σ)
    σ = exp(log_σ)
    llike = -n/2*log(2π) - n/2* log(σ^2) - (sum((Y - X * β).^2) / (2σ^2))
    llike = -llike
end

# The log likelihood function accepts 4 inputs: the matrix of
# explanatory variables (X), the dependent variable (Y), the β's, and
# the error varicance. Note that we exponentiate the error variance in
# the second line of the code because the error variance cannot be
# negative and we want to avoid this situation when maximizing the
# likelihood.
#
# The next step in our tutorial is to optimize our function. We first
# use the `TwiceDifferentiable` command in order to obtain the Hessian
# matrix later on, which will be used to help form t-statistics:

func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar + 1]),
                           ones(nvar+1); autodiff=:forward);

# The above statment accepts 4 inputs: the x matrix, the dependent
# variable y, and a vector of β's and the error variance.  The
# `vars[1:nvar]` is how we pass the vector of β's and the `vars[nvar +
# 1]` is how we pass the error variance. You can think of this as a
# vector of parameters with the first 2 being β's and the last one is
# the error variance.
#
# The `ones(nvar+1)` are the starting values for the parameters and
# the `autodiff=:forward` command performs forward mode automatic
# differentiation.
#
# The actual optimization of the likelihood function is accomplished
# with the following command:

opt = optimize(func, ones(nvar+1))

## Test the results                #src
using Test                    #src
@test Optim.converged(opt)         #src
@test Optim.g_residual(opt) < 1e-8 #src


# The first input to the command is the function we wish to optimize
# and the second input are the starting values.
#
# After a brief period of time, you should see output of the
# optimization routine, with the parameter  estimates being very close
# to our simulated values.
#
# The optimization routine stores several quantities and we can obtain
# the maximim likelihood estimates with the following command:

parameters = Optim.minimizer(opt)
@test parameters ≈ [2.83664, 3.05345, -0.98837] atol=1e-5 #src

# !!! Note
#     Fieldnames for all of the quantities can be obtained with the following command:
#     fieldnames(opt)
#

# In order to obtain the correct Hessian matrix, we have to "push" the
# actual parameter values that maximizes the likelihood function since
# the `TwiceDifferentiable` command uses the next to last values to
# calculate the Hessian:

numerical_hessian = hessian!(func,parameters)

# Let's find the estimated value of σ, rather than log σ, and it's standard error
# To do this, we will use the Delta Method: https://en.wikipedia.org/wiki/Delta_method 

# this function exponetiates log σ
function transform(parameters)
    parameters[end] = exp(parameters[end])
    parameters
end    

# get the Jacobian of the transformation
J = ForwardDiff.jacobian(transform, parameters)'
parameters = transform(parameters)

# We can now invert our Hessian matrix  and use the Delta Method,
# to obtain the variance-covariance matrix:
var_cov_matrix = J*inv(numerical_hessian)*J'

# test the estimated parameters and t-stats for correctness
@test parameters ≈ [2.83664, 3.05345, 0.37218] atol=1e-5 #src
t_stats = parameters./sqrt.(diag(var_cov_matrix))
@test t_stats ≈ [48.02655, 45.51568, 8.94427] atol=1e-4 #src

# see the results
println("parameter estimates:", parameters)
println("t-statsitics: ", t_stats)

# From here, one may examine other statistics of interest using the
# output from the optimization routine.

#md # ## [Plain Program](@id maxlikenlm-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [maxlikenlm.jl](maxlikenlm.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
