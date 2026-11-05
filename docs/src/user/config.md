# Configurable options
There are several options that simply take on some default values if the user
doesn't supply anything else than a function (and gradient) and a starting point.
## Solver options
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

## General Options
In addition to the solver, you can alter the behavior of the Optim package by using the list of keyword below in the `Optim.Options` constructor.

### Termination
* `x_abstol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `x_reltol`: Relative tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_abstol`: Absolute tolerance in changes of the objective value. Defaults to `0.0`.
* `f_reltol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_abstol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `newton_decrement_abstol`: Absolute tolerance on half the squared Newton decrement, `λ²/2 = gᵀB⁻¹g/2`, where `B` is the Hessian or the method's approximation of it. Defaults to `NaN` (off).
* `newton_decrement_reltol`: Relative tolerance on the same quantity: the run stops when `λ²/2 ≤ newton_decrement_reltol * |f(x)|`. Defaults to `NaN` (off).
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `true`. Note that, when this setting is `true`, the last iterate will be returned as the minimizer even if the objective increased. This is useful in some methods such as line search based methods with HagerZhang() that allows for a slight increase in the objective function as long as the directional derivative decreases. 
* `successive_f_tol`: Determines the number of times the objective is allowed to increase across iterations. Defaults to 1.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).
* `callback`: A function to be called after every iteration. The return value should be a boolean, where `true` will stop the `optimize` call early. The argument to the callback is the state of the optimization algorithm.

!!! note "The Newton decrement"
    `λ²/2` is the decrease that the method's quadratic model predicts for a full step,
    so it estimates `f(x) - f*` in the units of the objective. Unlike `|g(x)|`, it does
    not change when the variables `x` are rescaled. `newton_decrement_abstol` bounds this
    estimate in the units of `f`, so it is unaffected by adding a constant to `f`, but has
    to follow the scale of `f`. `newton_decrement_reltol` bounds it as a fraction of `|f(x)|`,
    so it is unaffected by scaling `f`, but it never fires on a problem whose minimum
    is zero, such as a least squares problem with zero residual. The run stops when
    either tolerance is met.

    The criterion is checked before each line search, once the search direction is
    known, so it costs no extra work. It is supported by `BFGS` and `LBFGS` once they
    hold curvature information from at least one step, by `NewtonTrustRegion` where
    the Hessian is positive definite, and by `KrylovTrustRegion` where its conjugate
    gradient solve stops on its residual test inside the trust region. The conjugate
    gradient estimate is smaller than `λ²/2`, by a relative error of at most `cg_tol²`
    times the condition number of the Hessian, so lower `cg_tol` on badly conditioned
    problems. `Fminbox` applies the criterion to the barrier problem of each inner run.
    Setting either tolerance for any other method throws an `ArgumentError`. The value
    at the returned point is `Optim.newton_decrement(res)`, which is `NaN` when both
    tolerances are off, and a run stopped by the criterion has the termination code
    `Optim.TerminationCode.NewtonDecrement`.

!!! warning "Quasi-Newton methods estimate the decrement"
    `BFGS` and `LBFGS` build `B` from past steps, so their `λ²/2` is only an estimate of
    the true decrement. Where `B` overstates the curvature at `x`, the estimate is too
    small, and the run can stop while `f(x) - f*` is still above the tolerance. This is
    likely where the curvature shrinks toward the minimizer, as it does when the Hessian
    is singular there. Set the tolerance below the accuracy you need in `f`.

!!! tip "Disabling a termination criterion"
    If the `x_abstol`, `x_reltol`, `f_abstol`, `f_reltol`, `g_tol`, or `time_limit` tolerances are set to `NaN` all comparisons will be false internally, and this fact can be used to turn off the check. For example, `x_reltol` defaults to `0`. This does not mean that the check is turned off it only means that we stop at exactly zero change. However, if we set it to `NaN` specifically, the check of the termination criterion is always false and as such we will never stop due to any value of the infinity norm of the vector of relative changes.


### Progress printing and storage
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `trace_variables`: A tuple of variable names given as `Symbol`s to store in the trace. Defaults to `(,)`, which means all variables are included.
* `show_warnings`: Should warnings due to NaNs or Inf be shown? Defaults to `true`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.

### Box constrained optimization (`Fminbox`)
Box constrained optimization has additional keywords to alter the behavior of the outer solver:

* `outer_x_abstol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `outer_x_reltol`: Relative tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `outer_f_abstol`: Absolute tolerance in changes of the objective value. Defaults to `0.0`.
* `outer_f_reltol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `outer_g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `allow_outer_f_increases`: Allow steps that increase the objective value. Defaults to `true`. Note that, when this setting is `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `outer_iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.

If you specify `outer_iterations = 10` and `iterations = 100`, the outer algorithm will run for `10` iterations, and for each outer iteration the inner algorithm will run for `100` iterations.

### Example usage
We currently recommend the statically dispatched interface by using the `Optim.Options`
constructor:
```julia-repl
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
```julia-repl
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
