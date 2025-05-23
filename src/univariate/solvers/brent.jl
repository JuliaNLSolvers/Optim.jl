"""
# Brent
## Constructor
```julia
    Brent(;)
```

## Description
Also known as the Brent-Dekker algorith, `Brent` is a univariate optimization
algorithm for minimizing functions on some interval `[a,b]`. The method uses bisection
to find a zero of the gradient. If the original interval contains a minimum,
bisection will reliably find the solution, but can be slow. To this end `Brent`
combines bisection with the secant method and inverse quadratic interpolation to
accelerate convergence.

## References
R. P. Brent (2002) Algorithms for Minimization Without Derivatives. Dover edition.
"""
struct Brent <: UnivariateOptimizer end

Base.summary(::Brent) = "Brent's Method"

function optimize(
    f,
    x_lower::T,
    x_upper::T,
    mo::Brent;
    rel_tol::T = sqrt(eps(T)),
    abs_tol::T = eps(T),
    iterations::Integer = 1_000,
    time_limit::Float64 = Inf,
    store_trace::Bool = false,
    show_trace::Bool = false,
    show_warnings::Bool = true,
    callback = nothing,
    show_every = 1,
    extended_trace::Bool = false,
) where {T<:AbstractFloat}
    t0 = time()
    options = (
        store_trace = store_trace,
        show_trace = show_trace,
        show_warnings = show_warnings,
        show_every = show_every,
        callback = callback,
        time_limit = time_limit,
    )
    if x_lower > x_upper
        error("x_lower must be less than x_upper")
    end

    # Save for later
    initial_lower = x_lower
    initial_upper = x_upper

    golden_ratio::T = T(1) / 2 * (3 - sqrt(T(5.0)))

    new_minimizer = x_lower + golden_ratio * (x_upper - x_lower)
    new_minimum = f(new_minimizer)
    best_bound = "initial"
    f_calls = 1 # Number of calls to f
    step = zero(T)
    old_step = zero(T)

    old_minimizer = new_minimizer
    old_old_minimizer = new_minimizer

    old_minimum = new_minimum
    old_old_minimum = new_minimum

    iteration = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace{T,typeof(mo)}()
    tracing = store_trace || show_trace || extended_trace || callback !== nothing
    stopped_by_callback = false
    if tracing
        # update trace; callbacks can stop routine early by returning true
        state = (
            new_minimizer = new_minimizer,
            x_lower = x_lower,
            x_upper = x_upper,
            best_bound = best_bound,
            new_minimum = new_minimum,
        )
        stopped_by_callback =
            trace!(tr, nothing, state, iteration, mo, options, time() - t0)
    end
    _time = time() - t0
    while iteration < iterations && !stopped_by_callback && _time < time_limit

        p = zero(T)
        q = zero(T)

        x_tol = rel_tol * abs(new_minimizer) + abs_tol

        x_midpoint = (x_upper + x_lower) / 2

        if abs(new_minimizer - x_midpoint) <= 2 * x_tol - (x_upper - x_lower) / 2
            converged = true
            break
        end

        iteration += 1

        if abs(old_step) > x_tol
            # Compute parabola interpolation
            # new_minimizer + p/q is the optimum of the parabola
            # Also, q is guaranteed to be positive

            r = (new_minimizer - old_minimizer) * (new_minimum - old_old_minimum)
            q = (new_minimizer - old_old_minimizer) * (new_minimum - old_minimum)
            p =
                (new_minimizer - old_old_minimizer) * q -
                (new_minimizer - old_minimizer) * r
            q = 2(q - r)

            if q > 0
                p = -p
            else
                q = -q
            end
        end

        if abs(p) < abs(q * old_step / 2) &&
           p < q * (x_upper - new_minimizer) &&
           p < q * (new_minimizer - x_lower)
            old_step = step
            step = p / q

            # The function must not be evaluated too close to x_upper or x_lower
            x_temp = new_minimizer + step
            if ((x_temp - x_lower) < 2 * x_tol || (x_upper - x_temp) < 2 * x_tol)
                step = (new_minimizer < x_midpoint) ? x_tol : -x_tol
            end
        else
            old_step =
                (new_minimizer < x_midpoint) ? x_upper - new_minimizer :
                x_lower - new_minimizer
            step = golden_ratio * old_step
        end

        # The function must not be evaluated too close to new_minimizer
        if abs(step) >= x_tol
            new_x = new_minimizer + step
        else
            new_x = new_minimizer + ((step > 0) ? x_tol : -x_tol)
        end

        new_f = f(new_x)
        f_calls += 1

        if new_f < new_minimum
            if new_x < new_minimizer
                x_upper = new_minimizer
                best_bound = "upper"
            else
                x_lower = new_minimizer
                best_bound = "lower"
            end
            old_old_minimizer = old_minimizer
            old_old_minimum = old_minimum
            old_minimizer = new_minimizer
            old_minimum = new_minimum
            new_minimizer = new_x
            new_minimum = new_f
        else
            if new_x < new_minimizer
                x_lower = new_x
            else
                x_upper = new_x
            end
            if new_f <= old_minimum || old_minimizer == new_minimizer
                old_old_minimizer = old_minimizer
                old_old_minimum = old_minimum
                old_minimizer = new_x
                old_minimum = new_f
            elseif new_f <= old_old_minimum ||
                   old_old_minimizer == new_minimizer ||
                   old_old_minimizer == old_minimizer
                old_old_minimizer = new_x
                old_old_minimum = new_f
            end
        end
        if tracing
            # update trace; callbacks can stop routine early by returning true
            state = (
                new_minimizer = new_minimizer,
                x_lower = x_lower,
                x_upper = x_upper,
                best_bound = best_bound,
                new_minimum = new_minimum,
            )
            stopped_by_callback =
                trace!(tr, nothing, state, iteration, mo, options, time() - t0)
        end
        _time = time() - t0
    end

    return UnivariateOptimizationResults(
        mo,
        initial_lower,
        initial_upper,
        new_minimizer,
        new_minimum,
        iteration,
        rel_tol,
        abs_tol,
        tr,
        f_calls,
        time_limit,
        _time,
        (; iterations = iteration == iterations, converged,)
        )
end


function trace!(tr, d, state, iteration, method::Brent, options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    dt["minimizer"] = state.new_minimizer
    dt["x_lower"] = state.x_lower
    dt["x_upper"] = state.x_upper
    dt["best bound"] = state.best_bound
    T = eltype(state.new_minimum)

    update!(
        tr,
        iteration,
        state.new_minimum,
        T(NaN),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
