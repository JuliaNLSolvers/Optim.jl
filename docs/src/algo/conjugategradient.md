# Conjugate Gradient Descent
## Constructor
```julia
ConjugateGradient(; linesearch! = hz_linesearch!,
                    eta = 0.4,
                    P = nothing,
                    precondprep! = (P, x) -> nothing)
```

## Description

## Example
## References
W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.
