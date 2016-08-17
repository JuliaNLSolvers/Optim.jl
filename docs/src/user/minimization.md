## Minimizing a multivariate function
To show how the Optim package can be used, we implement the
[Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function),
a classic problem in numerical optimization. We'll assume that you've already
installed the Optim package using Julia's package manager.
First, we load Optim and define the Rosenbrock function:
```jl
using Optim
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
```
Once we've defined this function, we can find the minimum of the Rosenbrock
function using any of our favorite optimization algorithms. With a function defined,
we just specify an initial point `x` and run:
```jl
optimize(f, [0.0, 0.0])
```
Optim will default to using the Nelder-Mead method in this case, as we did not provide a gradient. This can also
be explicitly specified using:
```jl
optimize(f, [0.0, 0.0], NelderMead())
```
Other solvers are available. Below, we use L-BFGS, a quasi-Newton method that requires a gradient.
If we pass `f` alone, Optim will construct an approximate gradient for us
using central finite differencing:
```jl
optimize(f, [0.0, 0.0], LBFGS())
```
Alternatively, the `autodiff` keyword will use atomatic differentiation to construct
the gradient.
```jl
optimize(f, [0.0, 0.0], LBFGS(), OptimizationOptions(autodiff = true))
```
For better performance and greater precision, you can pass your own gradient function. For the Rosenbrock example, the analytical gradient can be shown to be:
```jl
function g!(x::Vector, storage::Vector)
storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
storage[2] = 200.0 * (x[2] - x[1]^2)
end
```
Note that the functions we're using to calculate the gradient (and later the Hessian `h!`) of the Rosenbrock function mutate a fixed-sized storage array, which is passed as an additional argument called `storage`. By mutating a single array over many iterations, this style of function definition removes the sometimes considerable costs associated with allocating a new array during each call to the `g!` or `h!` functions. You can use `Optim` without manually defining a gradient or Hessian function, but if you do define these functions, they must take these two arguments in this order.
Returning to our optimization, you simply pass `g!` together with `f` from before to use the gradient:
```jl
optimize(f, g!, [0.0, 0.0], LBFGS())
```
For some methods, like simulated annealing, the gradient will be ignored:
```jl
optimize(f, g!, [0.0, 0.0], SimulatedAnnealing())
```
In addition to providing gradients, you can provide a Hessian function `h!` as well. In our current case this is:
```jl
function h!(x::Vector, storage::Matrix)
storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
storage[1, 2] = -400.0 * x[1]
storage[2, 1] = -400.0 * x[1]
storage[2, 2] = 200.0
end
```
Now we can use Newton's method for optimization by running:
```jl
optimize(f, g!, h!, [0.0, 0.0])
```
Which defaults to `Newton()` since a Hessian was provided. Like gradients, the Hessian function will be ignored if you use a method that does not require it:
```jl
optimize(f, g!, h!, [0.0, 0.0], LBFGS())
```
Note that Optim will not generate approximate Hessians using finite differencing
because of the potentially low accuracy of approximations to the Hessians. Other
than Newton's method, none of the algorithms provided by the Optim package employ
exact Hessians.

## Box minimization

A primal interior-point algorithm for simple "box" constraints (lower and upper bounds) is also available:

```jl
function f(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
results = optimize(DifferentiableFunction(f), initial_x, lower, upper, Fminbox(), optimizer = GradientDescent)
```

This performs optimization with a barrier penalty, successively scaling down the barrier coefficient and using the chosen `optimizer` for convergence at each step. Notice that the `Optimizer` type, not an instance should be passed. This means that the keyword should be passed as `optimizer = GradientDescent` not `optimizer = GradientDescent()`, as you usually would.

This algorithm uses diagonal preconditioning to improve the accuracy, and hence is a good example of how to use `ConjugateGradient` or `LBFGS` with preconditioning. Other methods will currently not use preconditioning. Only the box constraints are used. If you can analytically compute the diagonal of the Hessian of your objective function, you may want to consider writing your own preconditioner.

There are two iterations parameters: an outer iterations parameter used to control `Fminbox` and an inner iterations parameter used to control the inner optimizer. For this reason, the options syntax is a bit different from the rest of the package. All parameters regarding the outer iterations are passed as keyword arguments, and options for the interior optimizer is passed as an `OptimizationOptions` type using the keyword `optimizer_o`.

For example, the following restricts the optimization to 2 major iterations
```julia
results = optimize(DifferentiableFunction(f), initial_x, l, u, Fminbox(); optimizer = GradientDescent, iterations = 2)
```
In contrast, the following sets the maximum number of iterations for each `ConjugateGradient` optimization to 2
```julia
results = Optim.optimize(DifferentiableFunction(f), initial_x, l, u, Fminbox(); optimizer = GradientDescent, optimizer_o = OptimizationOptions(iterations = 2))
```
## Minimizing a univariate function on a bounded interval

Minimization of univariate functions without derivatives is available through
the `optimize` interface:

```jl
f_univariate(x) = 2x^2+3x+1
optimize(f_univariate, -2.0, 1.0)
```

Two methods are available:

* Brent's method, the default (can be explicitly selected with `Brent()`).
* Golden section search, available with `GoldenSection()`.

In addition to the `iterations`, `store_trace`, `show_trace` and
`extended_trace` options, the following options are also available:

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
Optim.method(res)
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

* `method(res)`
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
