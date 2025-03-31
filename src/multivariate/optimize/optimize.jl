update_g!(d, state, method) = nothing
function update_g!(d, state, method::FirstOrderOptimizer)
    # Update the function value and gradient
    value_gradient!(d, state.x)
    project_tangent!(method.manifold, gradient(d), state.x)
end
function update_g!(d, state, method::Newton)
    # Update the function value and gradient
    value_gradient!(d, state.x)
end
update_fg!(d, state, method) = nothing
update_fg!(d, state, method::ZerothOrderOptimizer) = value!(d, state.x)
function update_fg!(d, state, method::FirstOrderOptimizer)
    value_gradient!(d, state.x)
    project_tangent!(method.manifold, gradient(d), state.x)
end
function update_fg!(d, state, method::Newton)
    value_gradient!(d, state.x)
end

# Update the Hessian
update_h!(d, state, method) = nothing
update_h!(d, state, method::SecondOrderOptimizer) = hessian!(d, state.x)

after_while!(d, state, method, options) = nothing

function initial_convergence(d, state, method::AbstractOptimizer, initial_x, options)
    gradient!(d, initial_x)
    stopped = !isfinite(value(d)) || any(!isfinite, gradient(d))
    g_residual(d, state) <= options.g_abstol, stopped
end
function initial_convergence(d, state, method::ZerothOrderOptimizer, initial_x, options)
    false, false
end
function optimize(
    d::D,
    initial_x::Tx,
    method::M,
    options::Options{T,TCallback} = Options(; default_options(method)...),
    state = initial_state(method, options, d, initial_x),
) where {D<:AbstractObjective,M<:AbstractOptimizer,Tx<:AbstractArray,T,TCallback}

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit
    tr = OptimizationTrace{typeof(value(d)),typeof(method)}()
    tracing =
        options.store_trace ||
        options.show_trace ||
        options.extended_trace ||
        options.callback !== nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

    g_converged, stopped = initial_convergence(d, state, method, initial_x, options)
    converged = g_converged || stopped
    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    options.show_trace && print_header(method)
    _time = time()
    trace!(tr, d, state, iteration, method, options, _time - t0)
    ls_success::Bool = true
    while !converged && !stopped && iteration < options.iterations
        iteration += 1
        ls_success = !update_state!(d, state, method)
        if !ls_success
            break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        end
        if !(method isa NewtonTrustRegion)
            update_g!(d, state, method) # TODO: Should this be `update_fg!`?
        end
        x_converged, f_converged, g_converged, f_increased =
            assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol + 1 : 0
        converged = x_converged || g_converged || (counter_f_tol > options.successive_f_tol)
        if !(converged && method isa Newton) && !(method isa NewtonTrustRegion)
            update_h!(d, state, method) # only relevant if not converged
        end
        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback =
                trace!(tr, d, state, iteration, method, options, time() - t0)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        _time = time()
        stopped_by_time_limit = _time - t0 > options.time_limit
        f_limit_reached =
            options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached =
            options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached =
            options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) ||
           stopped_by_callback ||
           stopped_by_time_limit ||
           f_limit_reached ||
           g_limit_reached ||
           h_limit_reached
            stopped = true
        end

        if method isa NewtonTrustRegion
            # If the trust region radius keeps on reducing we need to stop
            # because something is wrong. Wrong gradients or a non-differentiability
            # at the solution could be explanations.
            if state.delta ≤ method.delta_min
                stopped = true
            end
        end

        if g_calls(d) > 0 && !all(isfinite, gradient(d))
            options.show_warnings && @warn "Terminated early due to NaN in gradient."
            break
        end
        if h_calls(d) > 0 && !(d isa TwiceDifferentiableHV) && !all(isfinite, hessian(d))
            options.show_warnings && @warn "Terminated early due to NaN in Hessian."
            break
        end
    end # while

    after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    Tf = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases
    stopped_by = (x_converged, f_converged, g_converged,
        f_limit_reached = f_limit_reached,
        g_limit_reached = g_limit_reached,
        h_limit_reached = h_limit_reached,
        time_limit = stopped_by_time_limit,
        callback = stopped_by_callback,
        f_increased = f_incr_pick,
        ls_failed = !ls_success,
        iterations = iteration == options.iterations,
    )

    termination_code =
        _termination_code(d, g_residual(d, state), state, stopped_by, options)

    return MultivariateOptimizationResults{
        typeof(method),
        Tx,
        typeof(x_abschange(state)),
        Tf,
        typeof(tr),
        typeof(stopped_by),
    }(
        method,
        initial_x,
        pick_best_x(f_incr_pick, state),
        pick_best_f(f_incr_pick, state, d),
        iteration,
        Tf(options.x_abstol),
        Tf(options.x_reltol),
        x_abschange(state),
        x_relchange(state),
        Tf(options.f_abstol),
        Tf(options.f_reltol),
        f_abschange(d, state),
        f_relchange(d, state),
        Tf(options.g_abstol),
        g_residual(d, state),
        tr,
        f_calls(d),
        g_calls(d),
        h_calls(d),
        options.time_limit,
        _time - t0,
        stopped_by,
        termination_code,
    )
end

function _termination_code(d, gres, state, stopped_by, options)

    if state isa NelderMeadState && gres <= options.g_abstol
        TerminationCode.NelderMeadCriterion
    elseif !(state isa NelderMeadState) && gres <= options.g_abstol
        TerminationCode.GradientNorm
    elseif (iszero(options.x_abstol) && x_abschange(state) <= options.x_abstol) ||
           (iszero(options.x_reltol) && x_relchange(state) <= options.x_reltol)
        TerminationCode.NoXChange
    elseif (iszero(options.f_abstol) && f_abschange(d, state) <= options.f_abstol) ||
           (iszero(options.f_reltol) && f_relchange(d, state) <= options.f_reltol)
        TerminationCode.NoObjectiveChange
    elseif x_abschange(state) <= options.x_abstol || x_relchange(state) <= options.x_reltol
        TerminationCode.SmallXChange
    elseif f_abschange(d, state) <= options.f_abstol ||
           f_relchange(d, state) <= options.f_reltol
        TerminationCode.SmallObjectiveChange
    elseif stopped_by.ls_failed
        TerminationCode.FailedLinesearch
    elseif stopped_by.callback
        TerminationCode.Callback
    elseif stopped_by.iterations
        TerminationCode.Iterations
    elseif stopped_by.time_limit
        TerminationCode.Time
    elseif stopped_by.f_limit_reached
        TerminationCode.ObjectiveCalls
    elseif stopped_by.g_limit_reached
        TerminationCode.GradientCalls
    elseif stopped_by.h_limit_reached
        TerminationCode.HessianCalls
    elseif stopped_by.f_increased
        TerminationCode.ObjectiveIncreased
    elseif f_calls(d) > 0 && !isfinite(value(d))
        TerminationCode.GradientNotFinite
    elseif g_calls(d) > 0 && !all(isfinite, gradient(d))
        TerminationCode.GradientNotFinite
    elseif h_calls(d) > 0 && !(d isa TwiceDifferentiableHV) && !all(isfinite, hessian(d))
        TerminationCode.HessianNotFinite
    else
        TerminationCode.NotImplemented
    end
end
