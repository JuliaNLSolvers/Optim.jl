# Optimization Functions for Julia

## Usage Examples

### Simple Function Demo

If you're just getting started, you probably want to use `optimize()`, which wraps the specific algorithms currently implemented and selects a good one based on the amount of information you can provide. See the examples below:

    using Optim

    eta = 0.9

    function f(x)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g(x)
      [x[1], eta * x[2]]
    end

    function h(x)
      [1.0 0.0; 0.0 eta]
    end

    # If you don't have gradient, uses Nelder-Mead.
    results = optimize(f, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

    # If you don't have Hessian, uses L-BFGS.
    results = optimize(f, g, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

    # If you have Hessian, uses Newton's method.
    results = optimize(f, g, h, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

Note that `optimize()` has some simple rules you must follow to use it effectively:

* The function to be optimized must take in Float64 vectors as input.
* The raw function `f` must return a scalar.
* The gradient `g` must return a vector.
* The Hessian `h` must return a matrix.

### Rosenbrock Function Demo

    using Optim

    function rosenbrock(x::Vector)
      (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end

    function rosenbrock_gradient(x::Vector)
      [-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1],
       200.0 * (x[2] - x[1]^2)]
    end

    function rosenbrock_hessian(x::Vector)
      h = zeros(2, 2)
      h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
      h[1, 2] = -400.0 * x[1]
      h[2, 1] = -400.0 * x[1]
      h[2, 2] = 200.0
      h
    end

    problem = Dict()
    problem[:f] = rosenbrock
    problem[:g] = rosenbrock_gradient
    problem[:h] = rosenbrock_hessian
    problem[:initial_x] = [0.0, 0.0]
    problem[:solution] = [1.0, 1.0]

    algorithms = ["naive_gradient_descent",
                  "gradient_descent",
                  "newton",
                  "bfgs",
                  "l-bfgs",
                  "nelder-mead",
                  "sa"]

    for algorithm = algorithms
      results = optimize(problem[:f],
                         problem[:g],
                         problem[:h],
                         problem[:initial_x],
                         algorithm,
                         10e-8,
                         true)
      print(results)
    end

### Curve Fitting Demo

There are also top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example:

    # a two-parameter exponential model
    model(xpts, p) = p[1]*exp(-xpts.*p[2])
    
    # some example data
    xpts = linspace(0,10,20)
    data = model(xpts, [1.0 2.0]) + 0.01*randn(length(xpts))
    
    beta, r, J = curve_fit(model, xpts, data, [0.5, 0.5])
	# beta = best fit parameters
	# r = vector of residuals
	# J = estimated Jacobian at solution
    
    # We can use these values to estimate errors on the fit parameters. To get 95% confidence error bars:
    errors = estimate_errors(beta, r, J)
    
## Conjugate gradients, box minimization, and nonnegative least squares

There is a separate suite of tools that implement the nonlinear conjugate gradient method, and there are some additional algorithms built on top of it. Unfortunately, currently these algorithms use a different API. These differences in API are intended to enhance performance. Rather than providing one function for the function value and another for the gradient, here you combine them into a single function. The function must be written to take the gradient as the first input. When the gradient is desired, that first input will be a vector; otherwise, the value `nothing` indicates that the gradient is not needed. Let's demonstrate this for the Rosenbrock Function:

    function rosenbrock(g, x::Vector)
      d1 = 1.0 - x[1]
      d2 = x[2] - x[1]^2
      if !(g === nothing)
        g[1] = -2.0*d1 - 400.0*d2*x[1]
        g[2] = 200.0*d2
      end
      val = d1^2 + 100.0 * d2^2
      return val
    end

In this example, you can see that we'll save a bit of time by not needing to recompute `d1` and `d2` in a separate gradient function.

More subtly, it does not require the allocation of a new vector to store the gradient; indeed, the conjugate-gradient algorithm reuses the same block of memory for the gradient on each iteration. While this design has substantial performance advantages, one common "gotcha" is overwriting the gradient array, for example by writing

    g = [-2.0*d1 - 400.0*d2*x[1], 200.0*d2]
    
Internally within `rosenbrock` this will appear to work, but the value is not passed back to the calling function and the memory locations for the original `g` may contain random values. Perhaps the easiest way to catch this type of error is to check, within `rosenbrock`, that the pointer is still the same as it was on entry:

      if !(g === nothing)
	gptr = pointer(g)
	# Code to assign values to g
        if pointer(g) != gptr
	  error("gradient vector overwritten")
	end
      end

### Conjugate gradient

The nonlinear conjugate gradient function is an implementation of an algorithm known as CG-DESCENT (see Citations below):

    x0 = [0.0,0.0]  # the initial guess
    x, fval, fcount, converged = cgdescent(rosenbrock, x0)

Here `x` is the solution vector, `fval` is a vector of function values after each iteration, `fcount` is the number of function evaluations, and `converged` is `true` if the algorithm converged to within the prescribed tolerance.

The algorithm can be controlled with a wide variety of options:

    using OptionsMod
    ops = @options display=Optim.ITER fcountmax=1000 tol=1e-5
    x, fval, fcount, converged = cgdescent(func, x0, ops)

This will cause it to display its progress at each iteration, limit itself to a maximum of 1000 function evaluations, and use a custom tolerance. There are many more options available, including a wide array of display options; for these, it's best to see the code.

### Box minimization

A primal interior-point algorithm for simple "box" constraints (lower and upper bounds) is also available:

    l = [1.25, -2.1]
    u = [Inf, Inf]
    x0 = [2.0, 2.0]
    x, fval, fcount, converged = fminbox(rosenbrock, x0, l, u, ops)

This performs optimization with a barrier penalty, successively scaling down the barrier coefficient and using `cgdescent` for convergence at each step.

This algorithm uses diagonal preconditioning to improve the accuracy, and hence is a good example of how to use `cgdescent` with preconditioning. Only the box constraints are used. If you can analytically compute the diagonal of the Hessian of your objective function, you may want to consider writing your own preconditioner (see `nnls` for an example).

### Nonnegative least-squares

Finally, one common application of box-constrained optimization is non-negative least-squares:

    A = randn(5,3)
    b = randn(size(A, 1))
    xnn, fval, fcount, converged = nnls(A, b)

This leverages fminbox and cgdescent; surely one could get even better performance by using an algorithm that takes advantage of this problem's linearity. Despite this, for large problems the performance is quite good compared to Matlab's `lsqnonneg`.

### Linear programming

For tools for doing linear programming, you should look into the MathProg package.

## State of the Library

### Existing Functions
* Constant Step-Size Gradient Descent: `naive_gradient_descent()`
* Back-Tracking Line Search Gradient Descent: `gradient_descent()`
* Guarded Newton's Method: `newton()`
* BFGS: `bfgs()`
* L-BFGS: `l_bfgs()`
* Nelder-Mead Method: `nelder_mead()`
* Simulated Annealing: `simulated_annealing()`
* Levenberg-Marquardt: `levenberg_marquardt()`
* Nonlinear conjugate-gradient: `cgdescent()`
* Box minimization: `fminbox()`
* Nonnegative least-squares: `nnls()`

### Planned Functions
* Brent's method
* Linear conjugate gradients
* L-BFGS-B (note that this functionality is already available in fminbox)

### Wrapping Functions
* Will provide methods for wrapping functions to insure they satisfy usage rules
* Will convert automatic conversion tools for input
* Will provide automatic differentiation

### Citations

W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.
