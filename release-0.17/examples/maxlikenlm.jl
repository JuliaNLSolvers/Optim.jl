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

using Optim, NLSolversBase, Random
using LinearAlgebra: diag
Random.seed!(0);                            # Fix random seed generator for reproducibility

#md # !!! tip
#md #     Add Optim with the following command at the Julia command prompt:
#md #     `Pkg.add("Optim")`
#
# The first item that needs to be addressed is the data generating process or DGP.
# The following code will produce data from a nomral linear model:


n = 500                             # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [ones(n) randn(n, nvar - 1)]    # X matrix of explanatory variables plus constant
ε = randn(n) * 0.5                  # Error variance
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
@test parameters ≈ [3.00279, 2.96455, -0.648693] atol=1e-5 #src

# !!! Note
#     Fieldnames for all of the quantities can be obtained with the following command:
#     fieldnames(opt)
#
# Since we paramaterized our likelihood to use the exponentiated
# value, we need to exponentiate it to get back to our original log
# scale:

parameters[nvar+1] = exp(parameters[nvar+1])

# In order to obtain the correct Hessian matrix, we have to "push" the
# actual parameter values that maximizes the likelihood function since
# the `TwiceDifferentiable` command uses the next to last values to
# calculate the Hessian:

numerical_hessian = hessian!(func,parameters)

# We can now invert our Hessian matrix to obtain the variance-covariance matrix:

var_cov_matrix = inv(numerical_hessian)

# In this example, we are only interested in the statistical
# significance of the coefficient estimates so we obtain those with
# the following command:

β = parameters[1:nvar]
@test β ≈ [3.00279, 2.96455] atol=1e-5 #src

# We now need to obtain those elements of the variance-covariance
# matrix needed to obtain our t-statistics, and we can do this with
# the following commands:

temp = diag(var_cov_matrix)
temp1 = temp[1:nvar]

# The t-statistics are formed by dividing element-by-element the
# coefficients by their standard errors, or the square root of the
# diagonal elements of the variance-covariance matrix:

t_stats = β./sqrt.(temp1)
@test t_stats ≈ [39.7191, 39.9506] atol=1e-4 #src

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
