Optim.jl
========

[![Optim](http://pkg.julialang.org/badges/Optim_0.3.svg)](http://pkg.julialang.org/?pkg=Optim&ver=0.3)
[![Optim](http://pkg.julialang.org/badges/Optim_0.4.svg)](http://pkg.julialang.org/?pkg=Optim&ver=0.4)

The Optim package represents an ongoing project to implement basic optimization algorithms in pure Julia under an MIT license. Because it is being developed from scratch, it is not as robust as the C-based NLOpt package. For work whose accuracy must be unquestionable, we recommend using the NLOpt package. See [the NLOpt.jl GitHub repository](https://github.com/stevengj/NLopt.jl) for details.

Although Optim is a work in progress, it is quite usable as is. In what follows, we describe the Optim package's API.

# Basic API Introduction

To show how the Optim package can be used, we'll implement the [Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function), a classic problem in numerical optimization. We'll assume that you've already installed the Optim package using Julia's package manager.

First, we'll load Optim and define the Rosenbrock function:

```jl
using Optim

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end
```

Note that the functions we're using to calculate the gradient and Hessian of the Rosenbrock function mutate a fixed-sized storage array, which is passed as an additional argument called `storage`. By mutating a single array over many iterations, this style of function definition removes the sometimes considerable costs associated with allocating a new array during each call to the `rosenbrock_gradient!` or `rosenbrock_hessian!` functions. You can use `Optim` without manually defining a gradient or Hessian function, but if you do define these functions, they must take these two arguments in this order.

Once we've defined these core functions, we can find the minimum of the Rosenbrock function using any of our favorite optimization algorithms. To make the code easier to read, we'll use shorter names for the core functions:

```jl
f = rosenbrock
g! = rosenbrock_gradient!
h! = rosenbrock_hessian!
```

With that done, the easiest way to perform optimization is to specify the core function `f` and an initial point, `x`:

```jl
optimize(f, [0.0, 0.0])
```
Optim will default to using the Nelder-Mead method in this case. We can specify the Nelder-Mead method explicitly using the `method` keyword:

```jl
optimize(f, [0.0, 0.0], method = :nelder_mead)
```

The `method` keyword also allows us to specify other methods as well. Below, we use L-BFGS, a quasi-Newton method that requires a gradient. If we pass `f` alone, Optim will construct an approximate gradient for us using central finite differencing:

```jl
optimize(f, [0.0, 0.0], method = :l_bfgs)
```

For greater precision, you should pass in the exact gradient function, `g!`:

```jl  
optimize(f, g!, [0.0, 0.0], method = :l_bfgs)
```

For some methods, like simulated annealing, the exact gradient will be ignored:

```jl
optimize(f, g!, [0.0, 0.0], method = :simulated_annealing)
```

In addition to providing exact gradients, you can provide an exact Hessian function `h!` as well:

```jl
optimize(f, g!, h!, [0.0, 0.0], method = :newton)
```
Like gradients, the Hessian function will be ignored if you use a method that does not require it:

```jl
    optimize(f, g!, h!, [0.0, 0.0], method = :l_bfgs)
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
res = optimize(sqerror, [0.0, 0.0], method = :l_bfgs)
```

# Configurable Options

The section above described the basic API for the Optim package. We employed several different optimization algorithms using the `method` keyword, which can take on any of the following values:

* `:bfgs`
* `:cg`
* `:gradient_descent`
* `:momentum_gradient_descent`
* `:l_bfgs`
* `:nelder_mead`
* `:newton`
* `:simulated_annealing`

In addition to the `method` keyword, you can alter the behavior of the Optim package by using the following keywords:


* `xtol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `ftol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `grtol`: What is the threshold for determining convergence? Defaults to `1e-8`.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `STDOUT`? Defaults to `false`.
* `autodiff`: When only an objective function is provided, use automatic differentiation to compute exact numerical gradients. If not, finite differencing will be used. This functionality is experimental. Defaults to `false`.

Thus, one might construct a complex call to `optimize` like:

```jl
res = optimize(f, g!,
               [0.0, 0.0],
               method = :gradient_descent,
               grtol = 1e-12,
               iterations = 10,
               store_trace = true,
               show_trace = false)
```

# Getting Better Performance

If you want to get better performance out of Optim, you'll need to dig into the internals. In particular, you'll need to understand the `DifferentiableFunction` and `TwiceDifferentiableFunction` types that the Optim package uses to couple a function `f` with its gradient `g!` and its Hessian `h!`. We could create objects of these types as follows:

```jl
d1 = DifferentiableFunction(rosenbrock)
d2 = DifferentiableFunction(rosenbrock,
                            rosenbrock_gradient!)
d3 = TwiceDifferentiableFunction(rosenbrock,
                                 rosenbrock_gradient!,
                                 rosenbrock_hessian!)
```

Note that `d1` above will use central finite differencing to approximate the gradient of `rosenbrock`.

In addition to these core ways of creating a `DifferentiableFunction` object, one can also create a `DifferentiableFunction` using three functions -- the third of which will evaluate the function and gradient simultaneously. To see this, let's implement such a joint evaluation function and insert it into a `DifferentiableFunction`:

```jl
function rosenbrock_and_gradient!(x::Vector, storage)
    d1 = (1.0 - x[1])
    d2 = (x[2] - x[1]^2)
    storage[1] = -2.0 * d1 - 400.0 * d2 * x[1]
    storage[2] = 200.0 * d2
    return d1^2 + 100.0 * d2^2
end

d4 = DifferentiableFunction(rosenbrock,
                            rosenbrock_gradient!,
                            rosenbrock_and_gradient!)
```

You can then use any of the functions contained in `d4` depending on performance/algorithm needs:

```jl
x = [0.0, 0.0]
y = d4.f(x)
storage = Array(Float64, length(x))
d4.g!(x, storage)
y = d4.fg!(x, storage)
```

If you do not provide a function like `rosenbrock_and_gradient!`, the constructor for `DifferentiableFunction` will define one for you automatically. By providing `rosenbrock_and_gradient!` function, you can sometimes get substantially better performance.

By defining a `DifferentiableFunction` that estimates function values and gradients simultaneously, you can sometimes achieve noticeable performance gains:

```jl
@elapsed optimize(f, g!, [0.0, 0.0], method = :bfgs)
@elapsed optimize(d4, [0.0, 0.0], method = :bfgs)
```

At the moment, the performance bottleneck for many problems is the simplistic backtracking line search we are using in Optim. As this step becomes more efficient, we expect that the gains from using a function that evaluates the main function and its gradient simultaneously will grow.
    
## Conjugate gradients, box minimization, and nonnegative least squares

There is a separate suite of tools that implement the nonlinear conjugate gradient method, and there are some additional algorithms built on top of it. Unfortunately, currently these algorithms use a different API. These differences in API are intended to enhance performance. Rather than providing one function for the function value and another for the gradient, here you combine them into a single function. The function must be written to take the gradient as the first input. When the gradient is desired, that first input will be a vector; otherwise, the value `nothing` indicates that the gradient is not needed. Let's demonstrate this for the Rosenbrock Function:

```jl
function rosenbrock(g, x::Vector)
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
    
Internally within `rosenbrock` this will appear to work, but the value is not passed back to the calling function and the memory locations for the original `g` may contain random values. Perhaps the easiest way to catch this type of error is to check, within `rosenbrock`, that the pointer is still the same as it was on entry:

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
results = fminbox(d4, x0, l, u)  # d4 from rosenbrock example
```

This performs optimization with a barrier penalty, successively scaling down the barrier coefficient and using `cg` for convergence at each step.

This algorithm uses diagonal preconditioning to improve the accuracy, and hence is a good example of how to use `cg` with preconditioning. Only the box constraints are used. If you can analytically compute the diagonal of the Hessian of your objective function, you may want to consider writing your own preconditioner (see `nnls` for an example).

### Nonnegative least-squares

Finally, one common application of box-constrained optimization is non-negative least-squares:

```jl
A = randn(5,3)
b = randn(size(A, 1))
results = nnls(A, b)
```

This leverages `fminbox` and `cg`; surely one could get even better performance by using an algorithm that takes advantage of this problem's linearity. Despite this, for large problems the performance is quite good compared to Matlab's `lsqnonneg`.

### Linear programming

For linear programming and extensions, see the [JuMP](https://github.com/JuliaOpt/JuMP.jl) and [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) packages.

## Univariate optimization without derivatives

Minimization of univariate functions without derivatives is available through
the `optimize` interface:

```jl
f(x) = 2x^2+3x+1
optimize(f, -2.0, 1.0)
```

Two methods are available:

* Brent's method, the default (can be explicitly selected with `method = :brent`).
* Golden section search, available with `method = :golden_section`.

In addition to the `iterations`, `store_trace`, `show_trace` and
`extended_trace` options, the following options are also available:

* `rel_tol`: The relative tolerance used for determining convergence. Defaults to `sqrt(eps(T))`.
* `abs_tol`: The absolute tolerance used for determining convergence. Defaults to `eps(T)`.

## State of the Library

### Existing Functions
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

### Planned Functions
* Linear conjugate gradients
* L-BFGS-B (note that this functionality is already available in fminbox)

### Citations

W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.

R. P. Brent (2002) Algorithms for Minimization Without Derivatives. Dover reedition.
