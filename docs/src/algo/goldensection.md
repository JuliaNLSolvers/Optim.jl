# Golden Section
```@docs
GoldenSection
```

## Description
Golden-section search reduces a finite scalar bracket without using derivatives.

## Example
```julia
using Optim
result = optimize(x -> (x - 2)^2, 0.0, 4.0, GoldenSection())
Optim.minimizer(result)
```
## References
