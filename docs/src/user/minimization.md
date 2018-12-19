## Unconstrained Optimization
To show how the Optim package can be used, we minimize the
[Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function),
a classical test problem for numerical optimization. We'll assume that you've already
installed the Optim package using Julia's package manager.
First, we load Optim and define the Rosenbrock function:
```jl
using Optim
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
```
Once we've defined this function, we can find the minimizer (the input that minimizes the objective) and the minimum (the value of the objective at the minimizer) using any of our favorite optimization algorithms. With a function defined,
we just specify an initial point `x` and call `optimize` with a starting point `x0`:
```jl
x0 = [0.0, 0.0]
optimize(f, x0)
```
*Note*: it is important to pass `initial_x` as an array. If your problem is one-dimensional, you have to wrap it in an array. An easy way to do so is to write `optimize(x->f(first(x)), [initial_x])` which make sure the input is an array, but the anonymous function automatically passes the first (and only) element onto your given `f`.

Optim will default to using the Nelder-Mead method in the multivariate case, as we did not provide a gradient. This can also
be explicitly specified using:
```jl
optimize(f, x0, NelderMead())
```
Other solvers are available. Below, we use L-BFGS, a quasi-Newton method that requires a gradient.
If we pass `f` alone, Optim will construct an approximate gradient for us using central finite differencing:
```jl
optimize(f, x0, LBFGS())
```
For better performance and greater precision, you can pass your own gradient function. If your objective is written in all Julia code with no special calls to external (that is non-Julia) libraries, you can also use automatic differentiation, by using the `autodiff` keyword and setting it to `:forward`:
```julia
optimize(f, x0, LBFGS(); autodiff = :forward)
```

For the Rosenbrock example, the analytical gradient can be shown to be:
```jl
function g!(G, x)
G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
G[2] = 200.0 * (x[2] - x[1]^2)
end
```
Note, that the functions we're using to calculate the gradient (and later the Hessian `h!`) of the Rosenbrock function mutate a fixed-sized storage array, which is passed as an additional argument called `G` (or `H` for the Hessian) in these examples. By mutating a single array over many iterations, this style of function definition removes the sometimes considerable costs associated with allocating a new array during each call to the `g!` or `h!` functions. If you prefer to have your gradients simply accept an `x`, you can still use `optimize` by setting the `inplace` keyword to `false`:
```jl
optimize(f, g, x0; inplace = false)
```
where `g` is a function of `x` only.

Returning to our in-place version, you simply pass `g!` together with `f` from before to use the gradient:
```jl
optimize(f, g!, x0, LBFGS())
```
For some methods, like simulated annealing, the gradient will be ignored:
```jl
optimize(f, g!, x0, SimulatedAnnealing())
```
In addition to providing gradients, you can provide a Hessian function `h!` as well. In our current case this is:
```jl
function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end
```
Now we can use Newton's method for optimization by running:
```jl
optimize(f, g!, h!, x0)
```
Which defaults to `Newton()` since a Hessian function was provided. Like gradients, the Hessian function will be ignored if you use a method that does not require it:
```jl
optimize(f, g!, h!, x0, LBFGS())
```
Note that Optim will not generate approximate Hessians using finite differencing
because of the potentially low accuracy of approximations to the Hessians. Other
than Newton's method, none of the algorithms provided by the Optim package employ
exact Hessians.

## Box Constrained Optimization

A primal interior-point algorithm for simple "box" constraints (lower and upper bounds) is available. Reusing our Rosenbrock example from above, boxed minimization is performed as follows:
```jl
lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
inner_optimizer = GradientDescent()
results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
```

This performs optimization with a barrier penalty, successively scaling down the barrier coefficient and using the chosen `inner_optimizer` (`GradientDescent()` above) for convergence at each step. To change algorithm specific options, such as the line search algorithm, specify it directly in the `inner_optimizer` constructor:
```
lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
# requires using LineSearches
inner_optimizer = GradientDescent(linesearch=LineSearches.BackTracking(order=3))
results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
```

This algorithm uses diagonal preconditioning to improve the accuracy, and hence is a good example of how to use `ConjugateGradient` or `LBFGS` with preconditioning. Other methods will currently not use preconditioning. Only the box constraints are used. If you can analytically compute the diagonal of the Hessian of your objective function, you may want to consider writing your own preconditioner.

There are two iterations parameters: an outer iterations parameter used to control `Fminbox` and an inner iterations parameter used to control the inner optimizer. For example, the following restricts the optimization to 2 major iterations
```julia
results = optimize(f, g!, lower, upper, initial_x, Fminbox(GradientDescent()), Optim.Options(outer_iterations = 2))
```
In contrast, the following sets the maximum number of iterations for each `ConjugateGradient()` optimization to 2
```julia
results = optimize(f, g!, lower, upper, initial_x, Fminbox(GradientDescent()), Optim.Options(iterations = 2))
```
## Minimizing a univariate function on a bounded interval

Minimization of univariate functions without derivatives is available through
the `optimize` interface:
```jl
optimize(f, lower, upper, method; kwargs...)
```
Notice the lack of initial `x`. A specific example is the following quadratic
function.
```jl
julia> f_univariate(x) = 2x^2+3x+1
f_univariate (generic function with 1 method)

julia> optimize(f_univariate, -2.0, 1.0)
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [-2.000000, 1.000000]
 * Minimizer: -7.500000e-01
 * Minimum: -1.250000e-01
 * Iterations: 7
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16): true
 * Objective Function Calls: 8
```
The output shows that we provided an initial lower and upper bound, that there is
a final minimizer and minimum, and that it used seven major iterations. Importantly,
we also see that convergence was declared. The default method is Brent's method,
which is one out of two available methods:

* Brent's method, the default (can be explicitly selected with `Brent()`).
* Golden section search, available with `GoldenSection()`.

If we want to manually specify this method, we use the usual syntax as for multivariate optimization.
```jl
    optimize(f, lower, upper, Brent(); kwargs...)
    optimize(f, lower, upper, GoldenSection(); kwargs...)
```

Keywords are used to set options for this special type of optimization. In addition to the `iterations`, `store_trace`, `show_trace` and `extended_trace` options, the following options are also available:

* `rel_tol`: The relative tolerance used for determining convergence. Defaults to `sqrt(eps(T))`.
* `abs_tol`: The absolute tolerance used for determining convergence. Defaults to `eps(T)`.

## Obtaining results
After we have our results in `res`, we can use the API for getting optimization results.
This consists of a collection of functions. They are not exported, so they have to be prefixed by `Optim.`.
Say we do the following optimization:
```jl
res = optimize(x->dot(x,[1 0. 0; 0 3 0; 0 0 1]*x), zeros(3))
```
 If we can't remember what method we used, we simply use
```jl
summary(res)
```
which will return `"Nelder Mead"`. A bit more useful information is the minimizer and minimum of the objective functions, which can be found using
```jlcon
julia> Optim.minimizer(res)
3-element Array{Float64,1}:
 -0.499921
 -0.3333
 -1.49994

julia> Optim.minimum(res)
 -2.8333333205768865
```

### Complete list of functions
A complete list of functions can be found below.

Defined for all methods:

* `summary(res)`
* `minimizer(res)`
* `minimum(res)`
* `iterations(res)`
* `iteration_limit_reached(res)`
* `trace(res)`
* `x_trace(res)`
* `f_trace(res)`
* `f_calls(res)`
* `converged(res)`

Defined for univariate optimization:

* `lower_bound(res)`
* `upper_bound(res)`
* `x_lower_trace(res)`
* `x_upper_trace(res)`
* `rel_tol(res)`
* `abs_tol(res)`

Defined for multivariate optimization:

* `g_norm_trace(res)`
* `g_calls(res)`
* `x_converged(res)`
* `f_converged(res)`
* `g_converged(res)`
* `initial_state(res)`

## Input types
Most users will input `Vector`'s as their `initial_x`'s, and get an `Optim.minimizer(res)` out that is also a vector. For zeroth and first order methods, it is also possible to pass in matrices, or even higher dimensional arrays. The only restriction imposed by leaving the `Vector` case is, that it is no longer possible to use finite difference approximations or automatic differentiation. Second order methods (variants of Newton's method) do not support this more general input type.

## Notes on convergence flags and checks
Currently, it is possible to access a minimizer using `Optim.minimizer(result)` even if
all convergence flags are `false`. This means that the user has to be a bit careful when using
the output from the solvers. It is advised to include checks for convergence if the minimizer
or minimum is used to carry out further calculations.

A related note is that first and second order methods makes a convergence check
on the gradient before entering the optimization loop. This is done to prevent
line search errors if `initial_x` is a stationary point. Notice, that this is only
a first order check. If `initial_x` is any type of stationary point, `g_converged`
will be true. This includes local minima, saddle points, and local maxima. If `iterations` is `0`
and `g_converged` is `true`, the user needs to keep this point in mind.
