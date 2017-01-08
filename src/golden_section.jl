macro goldensectiontrace()
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["minimizer"] = new_minimizer
                dt["x_lower"] = x_lower
                dt["x_upper"] = x_upper
            end
            update!(tr,
                    iteration,
                    new_minimum,
                    NaN,
                    dt,
                    store_trace,
                    show_trace,
                    show_every,
                    callback)
        end
    end)
end

immutable GoldenSection <: Optimizer end

function optimize{F<:Function, T <: AbstractFloat}(f::F, x_lower::T, x_upper::T,
                                      mo::GoldenSection;
                                      rel_tol::T = sqrt(eps(T)),
                                      abs_tol::T = eps(T),
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      callback = nothing,
                                      show_every = 1,
                                      extended_trace::Bool = false,
                                      nargs...)
    if !(x_lower < x_upper)
        error("x_lower must be less than x_upper")
    end

    # Save for later
    initial_lower = x_lower
    initial_upper = x_upper

    const golden_ratio::T = 0.5 * (3.0 - sqrt(5.0))

    new_minimizer = x_lower + golden_ratio*(x_upper-x_lower)
    new_minimum = f(new_minimizer)
    f_calls = 1 # Number of calls to f

    iteration = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace{typeof(mo)}()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @goldensectiontrace

    while iteration < iterations

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
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_upper = new_x
            end
        else
            new_x = new_minimizer - golden_ratio*(new_minimizer - x_lower)
            new_f = f(new_x)
            f_calls += 1
            if new_f < new_minimum
                x_upper = new_minimizer
                new_minimizer = new_x
                new_minimum = new_f
            else
                x_lower = new_x
            end
        end

        @goldensectiontrace
    end

    return UnivariateOptimizationResults("Golden Section Search",
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
