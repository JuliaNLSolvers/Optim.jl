## Planned Changes
Some features and changes have been identified as "wanted". Some of these are breaking
changes, but for the better.

One important change is the ordering of the preallocated array, and the current
iterate in the mutating gradient and Hessian functions. Currently, we have
```julia
g!(x, stor)
h!(x, stor)
```
But with the next version of Optim, we intend to be more in line with the rest
of the Julia ecosystem and write
```julia
g!(stor, x)
h!(stor, x)
```

Ít is also quite possible that the keywords-style tuning of options will be removed.
Instead of writing `optimize(..., g_tol = 1e-4)` users will have to write `optimize(..., OptimizationOptions(g_tol = 1e-4))`.
Obviously, it is a bit more verbose, but it will allow the internals to work entirely through
dispatch, and avoid a lot of keyword handling. Simpler code is easier to maintain,
and less prone to bugs.
