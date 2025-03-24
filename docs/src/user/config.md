## Configurable options
There are several options that simply take on some default values if the user
doesn't supply anything else than a function (and gradient) and a starting point.
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
In addition to the solver, you can alter the behavior of the Optim package by using the list of keyword below in the `Optim.Options` constructor. 

#### Termination
* `x_abstol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `x_reltol`: Relative tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_abstol`: Absolute tolerance in changes of the objective value. Defaults to `0.0`.
* `f_reltol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `true`. Note that, when this setting is `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `successive_f_tol`: Determines the number of times the objective is allowed to increase across iterations. Defaults to 1.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).
* `callback`: A function to be called during tracing. The return value should be a boolean, where `true` will stop the `optimize` call early. The callback function is called every `show_every`th iteration. If `store_trace` is false, the argument to the callback is of the type  [`OptimizationState`](https://github.com/JuliaNLSolvers/Optim.jl/blob/a1035134ca1f3ebe855f1cde034e32683178225a/src/types.jl#L155), describing the state of the current iteration. If `store_trace` is true, the argument is a list of all the states from the first iteration to the current.

!!! tip "Disabling a termination criterion
    If the `x_abstol`, `x_reltol`, `f_abstol`, `f_reltol`, `g_tol`, or `time_limit` tolerances are set to `NaN` all comparisons will be false internally, and this fact can be used to turn off the check. For example, `x_reltol` defaults to `0`. This does not mean that the check is turned off it only means that we stop at exactly zero change. However, if we set it to `NaN` specifically, the check of the termination criterion is always false and as such we will never stop due to any value of the infinity norm of the vector of relative changes.
    

#### Progress printing and storage
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `show_warnings`: Should warnings due to NaNs or Inf be shown? Defaults to `true`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.

#### Box constrained optimization (`Fminbox`)
Box constrained optimization has additional keywords to alter the behavior of the outer solver:

* `outer_x_abstol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `outer_x_reltol`: Relative tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `outer_f_abstol`: Absolute tolerance in changes of the objective value. Defaults to `0.0`.
* `outer_f_reltol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `outer_g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `allow_outer_f_increases`: Allow steps that increase the objective value. Defaults to `true`. Note that, when this setting is `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `outer_iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.

If you specify `outer_iterations = 10` and `iterations = 100`, the outer algorithm will run for `10` iterations, and for each outer iteration the inner algorithm will run for `100` iterations.

#### Example usage
We currently recommend the statically dispatched interface by using the `Optim.Options`
constructor:
```jl
res = optimize(f, g!,
               [0.0, 0.0],
               GradientDescent(),
               Optim.Options(g_tol = 1e-12,
                             iterations = 10,
                             store_trace = true,
                             show_trace = false,
                             show_warnings = true))
```
Another interface is also available, based directly on keywords:
```jl
res = optimize(f, g!,
               [0.0, 0.0],
               method = GradientDescent(),
               g_tol = 1e-12,
               iterations = 10,
               store_trace = true,
               show_trace = false,
               show_warnings = true)
```
Notice the need to specify the method using a keyword if this syntax is used.
This approach might be deprecated in the future, and as a result we recommend writing code
that has to maintained using the `Optim.Options` approach.
