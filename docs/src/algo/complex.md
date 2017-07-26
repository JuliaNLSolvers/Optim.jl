# Complex optimization
Optimization of functions defined on complex inputs is supported by simply passing a complex `x0` as input. All zeroth and first order optimization algorithms are supported.

The gradient of a complex-to-real function is defined as the only vector `g` such that `f(x+h) = f(x) + real(g' * h) + O(h^2)`. This is sometimes written `g = df/d(z*) = df/d(re(z)) + i df/d(im(z))`.

Because in general the gradient, even if differentiable as a function of R^2n, might not be complex-differentiable, the Hessian is not a well-defined concept and second-order optimization algorithms are not applicable directly. To use second-order optimization, convert to real variables. 
