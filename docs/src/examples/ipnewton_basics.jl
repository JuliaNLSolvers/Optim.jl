# # Nonlinear constrained optimization
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`ipnewton_basics.ipynb`](@__NBVIEWER_ROOT_URL__examples/generated/ipnewton_basics.ipynb)
#-
#
# The nonlinear constrained optimization interface in
# `Optim` assumes that the user can write the optimization
# problem in the following way.

# ```math
# \min_{x\in\mathbb{R}^n} f(x) \quad \text{such that}\\
# l_x \leq \phantom{c(}x\phantom{)} \leq u_x \\
# l_c \leq c(x) \leq u_c.
# ```

# For equality constraints on ``x_j`` or ``c(x)_j`` you set those
# particular entries of bounds to be equal, ``l_j=u_j``.
# Likewise, setting ``l_j=-\infty`` or ``u_j=\infty`` means that the
# constraint is unbounded from below or above respectively.

using Optim, NLSolversBase #hide
import NLSolversBase: clear! #hide

# # Constrained optimization with `IPNewton`

# We will go through examples on how to use the constraints interface
# with the interior-point Newton optimization algorithm [IPNewton](../../algo/ipnewton.md).

# Throughout these examples we work with the standard Rosenbrock function.
# The objective and its derivatives are given by


fun(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function fun_grad!(g, x)
g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
g[2] = 200.0 * (x[2] - x[1]^2)
end

function fun_hess!(h, x)
h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
h[1, 2] = -400.0 * x[1]
h[2, 1] = -400.0 * x[1]
h[2, 2] = 200.0
end;

# ## Optimization interface
# To solve a constrained optimization problem we call the `optimize`
# method
# ``` julia
# optimize(d::AbstractObjective, constraints::AbstractConstraints, initial_x::Tx, method::ConstrainedOptimizer, options::Options)
# ```

# We can create instances of `AbstractObjective` and
# `AbstractConstraints` using the types `TwiceDifferentiable` and
# `TwiceDifferentiableConstraints` from the package `NLSolversBase.jl`.


# ## Box minimzation
# We want to optimize the Rosenbrock function in the box
# ``-0.5 \leq x \leq 0.5``, starting from the point ``x_0=(0,0)``.
# Box constraints are defined using, for example,
# `TwiceDifferentiableConstraints(lx, ux)`.

x0 = [0.0, 0.0]
df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

lx = [-0.5, -0.5]; ux = [0.5, 0.5]
dfc = TwiceDifferentiableConstraints(lx, ux)

res = optimize(df, dfc, x0, IPNewton())
## Test the results             #src
using Test                 #src
@test Optim.converged(res)      #src
@test Optim.minimum(res) ≈ 0.25 #src

# If we only want to set lower bounds, use `ux = fill(Inf, 2)`

ux = fill(Inf, 2)
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                   #src
@test Optim.minimum(res) < 0.0 + sqrt(eps()) #src

# ## Defining "unconstrained" problems

# An unconstrained problem can be defined either by passing
# `Inf` bounds or empty arrays.
# **Note that we must pass the correct type information to the empty `lx` and `ux`**

lx = fill(-Inf, 2); ux = fill(Inf, 2)
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                   #src
@test Optim.minimum(res) < 0.0 + sqrt(eps()) #src

lx = Float64[]; ux = Float64[]
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                   #src
@test Optim.minimum(res) < 0.0 + sqrt(eps()) #src

# ## Generic nonlinear constraints

# We now consider the Rosenbrock problem with a constraint on
# ```math
#    c(x)_1 = x_1^2 + x_2^2.
# ```

# We pass the information about the constraints to `optimize`
# by defining a vector function `c(x)` and its Jacobian `J(x)`.

# The Hessian information is treated differently, by considering the
# Lagrangian of the corresponding slack-variable transformed
# optimization problem. This is similar to how the [CUTEst
# library](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) works.
# Let ``H_j(x)`` represent the Hessian of the ``j``th component
# ``c(x)_j`` of the generic constraints.
# and ``\lambda_j`` the corresponding dual variable in the
# Lagrangian. Then we want the `constraint` object to
# add the values of ``H_j(x)`` to the Hessian of the objective,
# weighted by ``\lambda_j``.

# The Julian form for the supplied function ``c(x)`` and the derivative
# information is then added in the following way.

con_c!(c, x) = (c[1] = x[1]^2 + x[2]^2; c)
function con_jacobian!(J, x)
    J[1,1] = 2*x[1]
    J[1,2] = 2*x[2]
    J
end
function con_h!(h, x, λ)
    h[1,1] += λ[1]*2
    h[2,2] += λ[1]*2
end;

# **Note that `con_h!` adds the `λ`-weighted Hessian value of each
# element of `c(x)` to the Hessian of `fun`.**


# We can then optimize the Rosenbrock function inside the ball of radius
# ``0.5``.

lx = Float64[]; ux = Float64[]
lc = [-Inf]; uc = [0.5^2]
dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                     lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                    #src
@test Optim.minimum(res) ≈ 0.2966215688829263 #src

# We can add a lower bound on the constraint, and thus
# optimize the objective on the annulus with
# inner and outer radii ``0.1`` and ``0.5`` respectively.

lc = [0.1^2]
dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                     lx, ux, lc, uc)
@suppress begin                               #src
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                    #src
@test Optim.minimum(res) ≈ 0.2966215688829255 #src
end                                           #src


# **Note that the algorithm warns that the Initial guess is not an
# interior point.** `IPNewton` can often handle this, however, if the
# initial guess is such that `c(x) = u_c`, then the algorithm currently
# fails. We may fix this in the future.


# ## Multiple constraints
# The following example illustrates how to add an additional constraint.
# In particular, we add a constraint function
# ```math
#    c(x)_2 = x_2\sin(x_1)-x_1
# ```

function con2_c!(c, x)
    c[1] = x[1]^2 + x[2]^2     ## First constraint
    c[2] = x[2]*sin(x[1])-x[1] ## Second constraint
    c
end
function con2_jacobian!(J, x)
    ## First constraint
    J[1,1] = 2*x[1]
    J[1,2] = 2*x[2]
    ## Second constraint
    J[2,1] = x[2]*cos(x[1])-1.0
    J[2,2] = sin(x[1])
    J
end
function con2_h!(h, x, λ)
    ## First constraint
    h[1,1] += λ[1]*2
    h[2,2] += λ[1]*2
    ## Second constraint
    h[1,1] += λ[2]*x[2]*-sin(x[1])
    h[1,2] += λ[2]*cos(x[1])
    ## Symmetrize h
    h[2,1]  = h[1,2]
    h
end;

# We generate the constraint objects and call `IPNewton` with
# initial guess ``x_0 = (0.25,0.25)``.

x0 = [0.25, 0.25]
lc = [-Inf, 0.0]; uc = [0.5^2, 0.0]
dfc = TwiceDifferentiableConstraints(con2_c!, con2_jacobian!, con2_h!,
                                     lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())
@test Optim.converged(res)                                       #src
@test Optim.minimum(res) ≈ 1.0                                   #src
@test isapprox(Optim.minimizer(res), zeros(2), atol=sqrt(eps())) #src

#md # ## [Plain Program](@id ipnewton_basics-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ipnewton_basics.jl](ipnewton_basics.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
