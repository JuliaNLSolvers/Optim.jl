## Configurable options
There are several options that simply take on some default values if the user
doensn't supply anything else than a function (and gradient) and a starting point.
### Solver options
There quite a few different solvers available in Optim, and they are all listed
below. Notice that the constructors are written without input here, but they
generally take keywords to tweak the way they work. See the pages describing each
solver for more detail.

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

Requires a function, a gradient, and a Hessian (cannot be omitted):

* `Newton()`

Box constrained minimization:

* `Fminbox()`

Special methods for univariate optimization:

* `Brent()`
* `GoldenSection()`

### General Options
In addition to the solver, you can alter the behavior of the Optim package by using the following keywords:

* `x_tol`: What is the threshold for determining convergence in the input vector? Defaults to `1e-32`.
* `f_tol`: What is the threshold for determining convergence in the objective value? Defaults to `1e-32`.
* `g_tol`: What is the threshold for determining convergence in the gradient? Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `STDOUT`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent.
* `autodiff`: When only an objective function is provided, use automatic differentiation to compute exact numerical gradients. If not, finite differencing will be used. This functionality is experimental. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.

We currently recommend the statically dispatched interface by using the `OptimizationOptions` 
constructor:
```jl
res = optimize(f, g!,
               [0.0, 0.0],
               GradientDescent(),
               OptimizationOptions(g_tol = 1e-12,
                                   iterations = 10,
                                   store_trace = true,
                                   show_trace = false))
```
Another interface is also available, based directly on keywords:
```jl
res = optimize(f, g!,
               [0.0, 0.0],
               method = GradientDescent(),
               g_tol = 1e-12,
               iterations = 10,
               store_trace = true,
               show_trace = false)
```
Notice the need to specify the method using a keyword if this syntax is used. It is likely that this will be deprecated in the future.
