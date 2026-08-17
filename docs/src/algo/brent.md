# Brent's Method
```@docs
Brent
```

## Description
Brent's method combines golden-section search and parabolic interpolation for
derivative-free minimization on a finite interval.

## Example
```julia
using Optim
result = optimize(x -> (x - 2)^2, 0.0, 4.0, Brent())
Optim.minimizer(result)
```
## References
R. P. Brent (2002) Algorithms for Minimization Without Derivatives. Dover edition.
