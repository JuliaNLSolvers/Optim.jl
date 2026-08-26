# Accelerated and Momentum Gradient Descent

`AcceleratedGradientDescent` and `MomentumGradientDescent` are first-order methods for
unconstrained objectives with an available gradient. Both use a line search and accept
the same `alphaguess`, `linesearch`, and `manifold` keyword arguments.

```@docs
AcceleratedGradientDescent
MomentumGradientDescent
```
