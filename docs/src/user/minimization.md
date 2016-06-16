## Minimizing a function
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
