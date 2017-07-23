# Complex optimization
Optimization of functions defined on complex inputs is supported by simply passing a complex `x0` as input. All zeroth and first order optimization algorithms are supported.

The gradient of a complex-to-real function is defined as the only vector `g` such that `f(x+h) = f(x) + real(g' * h) + O(h^2)`. This is sometimes written `g = df/d(z*) = df/d(re(z)) + i df/d(im(z))`.

Because in general the gradient is not a holomorphic function of `z`, the Hessian is not a well-defined concept and second-order optimization algorithms are not applicable directly. To use second-order optimization, convert to real variables. 

For first-order methods, preconditioners are assumed to be complex matrices, which is slightly restrictive for the reason mentioned above. If you need a more general operator, convert to real variables or create a custom type for P and overload `A_ldiv_B!(pg, P, g)` and `dot(x, P, y)`.
