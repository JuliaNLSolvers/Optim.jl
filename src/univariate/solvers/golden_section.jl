"""
# GoldenSection
## Constructor
```julia
    GoldenSection(;)
```

## Description
The `GoldenSection` method seeks to minimize a univariate function on an interval
`[a, b]`. At all times the algorithm maintains a tuple of three minimizer candidates
`(c, d, e)` where ``c<d<e`` such that the ratio of the largest to the smallest interval
is the Golden Ratio.

## References
https://en.wikipedia.org/wiki/Golden-section_search
"""
struct GoldenSection <: UnivariateOptimizer end

Base.summary(::GoldenSection) = "Golden Section Search"

function optimize(f, x_lower::T, x_upper::T,
     mo::GoldenSection;
     rel_tol::T = sqrt(eps(T)),
     abs_tol::T = eps(T),
     iterations::Integer = 1_000,
     store_trace::Bool = false,
     show_trace::Bool = false,
     callback = nothing,
     show_every = 1,
     extended_trace::Bool = false,
     nargs...) where T <: AbstractFloat
    if x_lower > x_upper
        error("x_lower must be less than x_upper")
    end
    t0 = time()
    options = (store_trace=store_trace, show_trace=show_trace, show_every=show_every, callback=callback)
    # Save for later
    initial_lower = x_lower
    initial_upper = x_upper

    golden_ratio::T = 0.5 * (3.0 - sqrt(5.0))

    new_minimizer = x_lower + golden_ratio*(x_upper-x_lower)
    new_minimum = f(new_minimizer)
    best_bound = "initial"
    f_calls = 1 # Number of calls to f

    iteration = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace{T, typeof(mo)}()
    tracing = store_trace || show_trace || extended_trace || callback !== nothing
    stopped_by_callback = false
    if tracing
        # update trace; callbacks can stop routine early by returning true
        state = (new_minimizer=new_minimizer,
                 x_lower=x_lower,
                 x_upper=x_upper,
                 best_bound=best_bound,
                 new_minimum=new_minimum)
        stopped_by_callback = trace!(tr, nothing, state, iteration, mo, options, time()-t0)
    end

    while iteration < iterations && !stopped_by_callback

        x_tol = rel_tol * abs(new_minimizer) + abs_tol

        x_midpoint = (x_upper+x_lower)/2

        if abs(new_minimizer - x_midpoint) <= 2*x_tol - (x_upper-x_lower)/2
            converged = true
            break
        end

        iteration += 1

        if x_upper - new_minimizer > new_minimizer - x_lower
            new_x = new_minimizer + golden_ratio*(x_upper - new_minimizer)
            new_f = f(new_x)
            f_calls += 1
            if new_f < new_minimum
                x_lower = new_minimizer
                best_bound = "lower"
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_upper = new_x
                best_bound = "upper"
            end
        else
            new_x = new_minimizer - golden_ratio*(new_minimizer - x_lower)
            new_f = f(new_x)
            f_calls += 1
            if new_f < new_minimum
                x_upper = new_minimizer
                best_bound = "upper"
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_lower = new_x
                best_bound = "lower"
            end
        end

        if tracing
            # update trace; callbacks can stop routine early by returning true
            state = (new_minimizer=new_minimizer,
                     x_lower=x_lower,
                     x_upper=x_upper,
                     best_bound=best_bound,
                     new_minimum=new_minimum)
            stopped_by_callback = trace!(tr, nothing, state, iteration, mo, options, time()-t0)
        end
    end

    return UnivariateOptimizationResults(mo,
                                         initial_lower,
                                         initial_upper,
                                         new_minimizer,
                                         new_minimum,
                                         iteration,
                                         iteration == iterations,
                                         converged,
                                         rel_tol,
                                         abs_tol,
                                         tr,
                                         f_calls)
end


function trace!(tr, d, state, iteration, method::GoldenSection, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    dt["minimizer"] = state.new_minimizer
    dt["x_lower"] = state.x_lower
    dt["x_upper"] = state.x_upper
    T = eltype(state.new_minimum)

    update!(tr,
            iteration,
            state.new_minimum,
            T(NaN),
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
