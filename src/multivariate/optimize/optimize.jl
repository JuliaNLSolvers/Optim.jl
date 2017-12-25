update_g!(d, state, method) = nothing
function update_g!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton}
    # Update the function value and gradient
    value_gradient!(d, state.x)
end
update_fg!(d, state, method) = nothing
update_fg!(d, state, method::ZerothOrderOptimizer) = value!(d, state.x)
update_fg!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton} = value_gradient!(d, state.x)

# Update the Hessian
update_h!(d, state, method) = nothing
update_h!(d, state, method::SecondOrderOptimizer) = hessian!(d, state.x)

after_while!(d, state, method, options) = nothing

function optimize(d::D, initial_x::AbstractArray, method::M,
    options::Options = Options(), state = initial_state(method, options, d, complex_to_real(d, initial_x))) where {D<:AbstractObjective, M<:AbstractOptimizer}
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("You cannot use NelderMead for univariate problems. Alternatively, use either interval bound univariate optimization, or another method such as BFGS or Newton.")
    end

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    initial_x = complex_to_real(d, initial_x)

    n = length(initial_x)
    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false

    g_converged = if typeof(method) <: NelderMead
        nmobjective(state.f_simplex, state.m, n) < options.g_tol
    elseif  typeof(method) <: ParticleSwarm || typeof(method) <: SimulatedAnnealing
        # TODO: remove KrylovTrustRegion when TwiceDifferentiableHV is in NLSolversBase
        false
    else
        gradient!(d, initial_x)
        vecnorm(gradient(d), Inf) < options.g_tol
    end

    converged = g_converged

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    options.show_trace && print_header(method)
    trace!(tr, d, state, iteration, method, options)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        update_state!(d, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS)
        update_g!(d, state, method)
        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, d, options)

        !converged && update_h!(d, state, method) # only relevant if not converged

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options)
        end

        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
        f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            stopped = true
        end
    end # while

    after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    elty = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases
    return MultivariateOptimizationResults(method,
                                        NLSolversBase.iscomplex(d),
                                        initial_x,
                                        pick_best_x(f_incr_pick, state),
                                        pick_best_f(f_incr_pick, state, d),
                                        iteration,
                                        iteration == options.iterations,
                                        x_converged,
                                        convert(elty, options.x_tol),
                                        x_abschange(state),
                                        f_converged,
                                        convert(elty, options.f_tol),
                                        f_abschange(d, state),
                                        g_converged,
                                        convert(elty, options.g_tol),
                                        g_residual(d),
                                        f_increased,
                                        tr,
                                        f_calls(d),
                                        g_calls(d),
                                        h_calls(d))
end
