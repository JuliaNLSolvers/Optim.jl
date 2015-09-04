macro goldensectiontrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x_minimum"] = x_minimum
                dt["x_lower"] = x_lower
                dt["x_upper"] = x_upper
            end
            update!(tr,
                    it,
                    f_minimum,
                    NaN,
                    dt,
                    store_trace,
                    show_trace,
                    show_every,
                    callback)
        end
    end
end

function golden_section{T <: FloatingPoint}(f::Function, x_lower::T, x_upper::T;
                                            rel_tol::T = sqrt(eps(T)),
                                            abs_tol::T = eps(T),
                                            iterations::Integer = 1_000,
                                            store_trace::Bool = false,
                                            show_trace::Bool = false,
                                            callback = nothing,
                                            show_every = 1,
                                            extended_trace::Bool = false)
    if !(x_lower < x_upper)
        error("x_lower must be less than x_upper")
    end

    # Save for later
    initial_lower = x_lower
    initial_upper = x_upper

    const golden_ratio::T = 0.5 * (3.0 - sqrt(5.0))

    x_minimum = x_lower + golden_ratio*(x_upper-x_lower)
    f_minimum = f(x_minimum)
    f_calls = 1 # Number of calls to f

    it = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @goldensectiontrace

    while it < iterations

        tolx = rel_tol * abs(x_minimum) + abs_tol

        x_midpoint = (x_upper+x_lower)/2

        if abs(x_minimum - x_midpoint) <= 2*tolx - (x_upper-x_lower)/2
            converged = true
            break
        end

        it += 1

        if x_upper - x_minimum > x_minimum - x_lower
            x_new = x_minimum + golden_ratio*(x_upper - x_minimum)
            f_new = f(x_new)
            f_calls += 1
            if f_new < f_minimum
                x_lower = x_minimum
                x_minimum = x_new
                f_minimum = f_new
            else
                x_upper = x_new
            end
        else
            x_new = x_minimum - golden_ratio*(x_minimum - x_lower)
            f_new = f(x_new)
            f_calls += 1
            if f_new < f_minimum
                x_upper = x_minimum
                x_minimum = x_new
                f_minimum = f_new
            else
                x_lower = x_new
            end
        end

        @goldensectiontrace
    end

    return UnivariateOptimizationResults("Golden Section Search",
                                         initial_lower,
                                         initial_upper,
                                         x_minimum,
                                         @compat(Float64(f_minimum)),
                                         it,
                                         converged,
                                         rel_tol,
                                         abs_tol,
                                         tr,
                                         f_calls)
end
