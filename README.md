*This it the development branch of Optim.jl. Please visit [this branch](https://github.com/JuliaOpt/Optim.jl/tree/v0.4.5) to find the README.md belonging to the latest official release of Optim.jl*

Optim.jl
========

[![Join the chat at https://gitter.im/JuliaOpt/Optim.jl](https://badges.gitter.im/JuliaOpt/Optim.jl.svg)](https://gitter.im/JuliaOpt/Optim.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Optim](http://pkg.julialang.org/badges/Optim_0.3.svg)](http://pkg.julialang.org/?pkg=Optim&ver=0.3)
[![Optim](http://pkg.julialang.org/badges/Optim_0.4.svg)](http://pkg.julialang.org/?pkg=Optim&ver=0.4)

The Optim package represents an ongoing project to implement basic optimization algorithms in pure Julia under an MIT license. Because it is being developed from scratch, it is not as robust as the C-based NLOpt package. For work whose accuracy must be unquestionable, we recommend using the NLOpt package. See [the NLOpt.jl GitHub repository](https://github.com/stevengj/NLopt.jl) for details.

Although Optim is a work in progress, it is quite usable as is. In what follows, we describe the Optim package's API.

# Basic API Introduction

To show how the Optim package can be used, we'll implement the [Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function), a classic problem in numerical optimization. We'll assume that you've already installed the Optim package using Julia's package manager.

First, we'll load Optim and define the Rosenbrock function:

```jl
using Optim

function f(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
```

Once we've defined this functions, we can find the minimum of the Rosenbrock function using any of our favorite optimization algorithms. With that done, we can just specify an initial point `x` and do:

```jl
optimize(f, [0.0, 0.0])
```
Optim will default to using the Nelder-Mead method in this case. This can also be explicitly specified using:

```jl
optimize(f, [0.0, 0.0], NelderMead())
```

The `method` keyword also allows us to specify other methods as well. Below, we use L-BFGS, a quasi-Newton method that requires a gradient. If we pass `f` alone, Optim will construct an approximate gradient for us using central finite differencing:

```jl
optimize(f, [0.0, 0.0], LBFGS())
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

For some methods, like simulated annealing, the exact gradient will be ignored:

```jl
optimize(f, g!, [0.0, 0.0], SimulatedAnnealing())
```

In addition to providing exact gradients, you can provide an exact Hessian function `h!` as well. In our current case this is:
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
optimize(f, g!, h!, [0.0, 0.0], Newton())
```
Like gradients, the Hessian function will be ignored if you use a method that does not require it:

```jl
optimize(f, g!, h!, [0.0, 0.0], LBFGS())
```

Note that Optim will not generate approximate Hessians using finite differencing because of the potentially low accuracy of approximations to the Hessians. Other than Newton's method, none of the algorithms provided by the Optim package employ exact Hessians.

# Optimizing Functions that Depend on Constant Parameters

In various fields, one sometimes needs to optimize a function that depends upon a set of parameters that are effectively constant with respect to the optimization procedure.

For example, in statistical computing, one frequently needs to optimize a "likelihood" function that depends on both (a) a set of model parameters and (a) a set of observed data points. As far as the `optimize` function is concerned, all function arguments are not constants, so one needs to define a specialized function that has all of the constants hardcoded into it.

We can do this using closures. For example, suppose that the observed data is found in the variables `x` and `y`:

```jl
x = [1.0, 2.0, 3.0]
y = 1.0 + 2.0 * x + [-0.3, 0.3, -0.1]
```

With the `x` and `y` variables present in the current scope, we can define a closure that is aware of the observed data, but depends only on the model parameters:

```jl
function sqerror(betas)
    err = 0.0
    for i in 1:length(x)
        pred_i = betas[1] + betas[2] * x[i]
        err += (y[i] - pred_i)^2
    end
    return err
end
```

We can then optimize the `sqerror` function just like any other function:

```jl
res = optimize(sqerror, [0.0, 0.0], LBFGS())
```

# API for accessing results

After we have our results in `res`, we can use the API for getting optimization results. This consists of a collection of functions. They are not exported, so they have to be prefixed by `Optim.`. Say we have optimized the `sqerror` function above. If we can't remember what method we used, we simply use
```jl
Optim.method(res)
```
which will return `"L-BFGS"`. A bit more useful information is the minimizer and minimum of the objective functions, which can be found using
```jl
Optim.minimizer(res)
# returns [0.766667, 2.1]     

Optim.minimum(res)
# returns 0.16666666666666652
```

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

# Configurable Options

The section above described the basic API for the Optim package, although it is on the roadmap to update this soon. We employed several different optimization algorithms using the `method` keyword, which can take on any of the following values:

Requires only a function handle:
* `NelderMead()`
* `SimulatedAnnealing()`

Requires a function and gradient (will be approximated if omitted):
* `BFGS()`
* `LBFGS()`
* `ConjugateGradient()`
* `GradientDescent()`
* `MomentumGradientDescent()`
* `AcceleratedGradientDescent()`

Requires a function, a gradient, and a hessian (cannot be omitted):
* `Newton()`

Box constrained minimization:
* `Fminbox()`

Special methods for univariate optimization:
* `Brent()`
* `GoldenSection()`

In addition to the `method` keyword, you can alter the behavior of the Optim package by using the following keywords:

* `xtol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `ftol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `grtol`: What is the threshold for determining convergence? Defaults to `1e-8`.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `STDOUT`? Defaults to `false`.
* `extended_trace`: Also save the current `x` and the gradient at `x`.
* `autodiff`: When only an objective function is provided, use automatic differentiation to compute exact numerical gradients. If not, finite differencing will be used. This functionality is experimental. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.

Thus, one might construct a complex call to `optimize` like:

```jl
res = optimize(f, g!,
               [0.0, 0.0],
               method = GradientDescent(),
               grtol = 1e-12,
               iterations = 10,
               store_trace = true,
               show_trace = false)
```

Notice the need to specify the method using a keyword if this syntax is used. It is also possible to call the statically dispatched interface directly using `OptimizationOptions`:

```jl
res = optimize(f, g!,
               [0.0, 0.0],
               GradientDescent(),
               OptimizationOptions(grtol = 1e-12,
                                   iterations = 10,
                                   store_trace = true,
                                   show_trace = false))
```

# Getting Better Performance

If you want to get better performance out of Optim, you'll need to dig into the internals. In particular, you'll need to understand the `DifferentiableFunction` and `TwiceDifferentiableFunction` types that the Optim package uses to couple a function `f` with its gradient `g!` and its Hessian `h!`. We could create objects of these types as follows:

```jl
d1 = DifferentiableFunction(f)
d2 = DifferentiableFunction(f,
                            g!)
d3 = TwiceDifferentiableFunction(f,
                                 g!,
                                 h!)
```

Note that `d1` above will use central finite differencing to approximate the gradient of `f`.

In addition to these core ways of creating a `DifferentiableFunction` object, one can also create a `DifferentiableFunction` using three functions -- the third of which will evaluate the function and gradient simultaneously. To see this, let's implement such a joint evaluation function and insert it into a `DifferentiableFunction`:

```jl
function fg!(x::Vector, storage)
    d1 = (1.0 - x[1])
    d2 = (x[2] - x[1]^2)
    storage[1] = -2.0 * d1 - 400.0 * d2 * x[1]
    storage[2] = 200.0 * d2
    return d1^2 + 100.0 * d2^2
end

d4 = DifferentiableFunction(f,
                            g!,
                            fg!)
```

You can then use any of the functions contained in `d4` depending on performance/algorithm needs:

```jl
x = [0.0, 0.0]
y = d4.f(x)
storage = Array(Float64, length(x))
d4.g!(x, storage)
y = d4.fg!(x, storage)
```

If you do not provide a function like `fg!`, the constructor for `DifferentiableFunction` will define one for you automatically. By providing `fg!` function, you can sometimes get substantially better performance.

By defining a `DifferentiableFunction` that estimates function values and gradients simultaneously, you can sometimes achieve noticeable performance gains:

```jl
@elapsed optimize(f, g!, [0.0, 0.0], BFGS())
@elapsed optimize(d4, [0.0, 0.0], BFGS())
```

At the moment, the performance bottleneck for many problems is the simplistic backtracking line search we are using in Optim. As this step becomes more efficient, we expect that the gains from using a function that evaluates the main function and its gradient simultaneously will grow.

## Conjugate gradients, box minimization, and nonnegative least squares

There is a separate suite of tools that implement the nonlinear conjugate gradient method, and there are some additional algorithms built on top of it. Unfortunately, currently these algorithms use a different API. These differences in API are intended to enhance performance. Rather than providing one function for the function value and another for the gradient, here you combine them into a single function. The function must be written to take the gradient as the first input. When the gradient is desired, that first input will be a vector; otherwise, the value `nothing` indicates that the gradient is not needed. Let's demonstrate this for the Rosenbrock Function:

```jl
function rosenbrock!(g, x::Vector)
  d1 = 1.0 - x[1]
  d2 = x[2] - x[1]^2
  if !(g === nothing)
    g[1] = -2.0*d1 - 400.0*d2*x[1]
    g[2] = 200.0*d2
  end
  val = d1^2 + 100.0 * d2^2
  return val
end
```

In this example, you can see that we'll save a bit of time by not needing to recompute `d1` and `d2` in a separate gradient function.

More subtly, it does not require the allocation of a new vector to store the gradient; indeed, the conjugate-gradient algorithm reuses the same block of memory for the gradient on each iteration. While this design has substantial performance advantages, one common "gotcha" is overwriting the gradient array, for example by writing

```
g = [-2.0*d1 - 400.0*d2*x[1], 200.0*d2]
```

Internally within `rosenbrock!` this will appear to work, but the value is not passed back to the calling function and the memory locations for the original `g` may contain random values. Perhaps the easiest way to catch this type of error is to check, within `rosenbrock!`, that the pointer is still the same as it was on entry:

```jl
if !(g === nothing)
	gptr = pointer(g)
	# Code to assign values to g
        if pointer(g) != gptr
	  error("gradient vector overwritten")
	end
end
```


### Box minimization

A primal interior-point algorithm for simple "box" constraints (lower and upper bounds) is also available:

```jl
l = [1.25, -2.1]
u = [Inf, Inf]
x0 = [2.0, 2.0]
results = optimize(d4, x0, l, u, Fminbox())  # d4 from rosenbrock example
```

This performs optimization with a barrier penalty, successively scaling down the barrier coefficient and using `ConjugateGradient` for convergence at each step.

This algorithm uses diagonal preconditioning to improve the accuracy, and hence is a good example of how to use `ConjugateGradient` with preconditioning. Only the box constraints are used. If you can analytically compute the diagonal of the Hessian of your objective function, you may want to consider writing your own preconditioner.

There are two iterations parameters: an outer iterations parameter used to control `Fminbox` and an inner iterations parameter used to control `ConjugateGradient`. For this reason, the options syntax is a bit different from the rest of the package. All parameters regarding the outer iterations are passed as keyword arguments, and options for the interior optimizer is passed as an `OptimizationOptions` type using the keyword `optimizer_o`.

For example, the following restricts the optimization to 2 major iterations
```julia
results = optimize(objective, x0, l, u, Fminbox(); iterations = 2)
```
In contrast, the following sets the maximum number of iterations for each `ConjugateGradient` optimization to 2
```julia
results = Optim.optimize(objective, x0, l, u, Fminbox(); optimizer_o = OptimizationOptions(iterations = 2))
```
### Linear programming

For linear programming and extensions, see the [JuMP](https://github.com/JuliaOpt/JuMP.jl) and [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) packages.

## Univariate optimization without derivatives

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

## Preconditioning

The `GradientDescent`, `ConjugateGradient` and `LBFGS` methods support preconditioning. A preconditioner
can be thought of as a change of coordinates under which the Hessian is better conditioned. With a
"good" preconditioner substantially improved convergence is possible.

An example of this is shown below (`Optimizer` ∈ {`GradientDescent`, `ConjugateGradient`, `LBFGS`}).
```jl
using ForwardDiff
plap(U; n=length(U)) = (n-1) * sum( (0.1 + diff(U).^2).^2 ) - sum(U) / (n-1)
plap1 = ForwardDiff.gradient(plap)
precond(n) = spdiagm( ( -ones(n-1), 2*ones(n), -ones(n-1) ), (-1,0,1), n, n) * (n+1)
df = DifferentiableFunction( X->plap([0;X;0]),
                             (X, G)->copy!(G, (plap1([0;X;0]))[2:end-1]) )
result = Optim.optimize(df, zeros(100), method=ConjugateGradient(P = nothing) )
result = Optim.optimize(df, zeros(100), method=ConjugateGradient(P = precond(100)) )
```
Benchmarking shows that using preconditioning provides an approximate speedup factor of 15 in this case.

The optimizers then use `precondprep!` to update the preconditioner after each update of the
state `x`. Further, to apply the preconditioner, they employ the the following three methods:
* `pprecondfwd!(out, P, A)` : apply `P` to a vector `A` and store in `out`
* `precondfwddot(A, P, B)` : take the inner product between `B` and `pprecondfwd!(out, P, A)`
* `precondinvdot(A, P, B)` : the dual inner product

Precisely what these operations mean, depends on how `P` is stored. Commonly, we store a matrix `P` which
approximates the Hessian in some vague sense. In this case,
* `pprecondfwd!(out, P, A) = copy!(out, P \ A)`
* `precondfwddot(A, P, B) = dot(A, P \ B)`
* `precondinvdot(A, P, B) = dot(A, P * B)`

# State of the Library

The current API calls for the user to use the `optimize` function with the appropriate `method` as shown above. Below is the old (deprecated) syntax.

## Existing Functions (deprecated)
* Gradient Descent: `gradient_descent()`
* Newton's Method: `newton()`
* BFGS: `bfgs()`
* L-BFGS: `l_bfgs()`
* Conjugate Gradient: `cg()`
* Nelder-Mead Method: `nelder_mead()`
* Simulated Annealing: `simulated_annealing()`
* Levenberg-Marquardt: `levenberg_marquardt()`
* Nonlinear conjugate-gradient: `cgdescent()`
* Box minimization: `fminbox()`
* Nonnegative least-squares: `nnls()`
* Brent's method: `brent()`
* Golden Section search: `golden_section()`

## Planned Functions
* Linear conjugate gradients
* L-BFGS-B (note that this functionality is already available in fminbox)

# Citations

W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.

R. P. Brent (2002) Algorithms for Minimization Without Derivatives. Dover reedition.
