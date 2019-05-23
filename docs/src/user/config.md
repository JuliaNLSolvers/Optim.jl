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
* `NewtonTrustRegion()`

Box constrained minimization:

* `Fminbox()`

Special methods for bounded univariate optimization:

* `Brent()`
* `GoldenSection()`

### General Options
In addition to the solver, you can alter the behavior of the Optim package by using the following keywords:

* `x_tol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_tol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `false`. Note that, when setting this to `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `callback`: A function to be called during tracing. A return value of `true` stops the `optimize` call.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).

We currently recommend the statically dispatched interface by using the `Optim.Options`
constructor:
```jl
res = optimize(f, g!,
               [0.0, 0.0],
               GradientDescent(),
               Optim.Options(g_tol = 1e-12,
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
Notice the need to specify the method using a keyword if this syntax is used.
This approach might be deprecated in the future, and as a result we recommend writing code
that has to maintained using the `Optim.Options` approach.
