## Obtaining results
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
    
