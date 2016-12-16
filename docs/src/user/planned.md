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
