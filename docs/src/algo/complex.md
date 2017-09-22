# Complex optimization
Optimization of functions defined on complex inputs (C^n to R) is supported by simply passing a complex `x0` as input. All zeroth and first order optimization algorithms are supported. For now, only explicit gradients are supported.

The gradient of a complex-to-real function is defined as the only vector `g` such that `f(x+h) = f(x) + real(g' * h) + O(h^2)`. This is sometimes written `g = df/d(z*) = df/d(re(z)) + i df/d(im(z))`.

The gradient of a C^n to R function is a C^n to C^n map. Even if it is differentiable when seen as a function of R^2n to R^2n, it might not be complex-differentiable. For instance, take f(z) = Re(z)^2. Then g(z) = 2 Re(z), which is not complex-differentiable (holomorphic). Therefore, the Hessian of a C^n to R function is in general not well-defined as a n x n complex matrix (only as a 2n x 2n real matrix), and therefore second-order optimization algorithms are not applicable directly. To use second-order optimization, convert to real variables. 
