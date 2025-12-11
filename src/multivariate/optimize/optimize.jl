# Update function value, gradient and Hessian
function update_fgh!(d, state, ::ZerothOrderOptimizer)
    f_x = NLSolversBase.value!(d, state.x)
    state.f_x = f_x
    return nothing
end
function update_fgh!(d, state, method::FirstOrderOptimizer)
    f_x, g_x = NLSolversBase.value_gradient!(d, state.x)
    copyto!(state.g_x, g_x)
    if hasproperty(method, :manifold)
        project_tangent!(method.manifold, state.g_x, state.x)
    end
    state.f_x = f_x
    return nothing
end
function update_fgh!(d, state, method::SecondOrderOptimizer)
    # Manifold optimization is currently not supported for second order optimization algorithms
    @assert !hasproperty(method, :manifold)

    f_x, g_x, H_x = NLSolversBase.value_gradient_hessian!(d, state.x)
    state.f_x = f_x
    copyto!(state.g_x, g_x)
    copyto!(state.H_x, H_x)

    return nothing
end

after_while!(d, state, method, options) = nothing

function initial_convergence(state::AbstractOptimizerState, options::Options)
    stopped = !isfinite(state.f_x) || any(!isfinite, state.g_x)
    return g_residual(state) <= options.g_abstol, stopped
end
function initial_convergence(::ZerothOrderState, ::Options)
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
    tr = OptimizationTrace{typeof(state.f_x),typeof(method)}()
    tracing =
        options.store_trace ||
        options.show_trace ||
        options.extended_trace ||
        options.callback !== nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0
    small_trustregion_radius = false

    g_converged, stopped = initial_convergence(state, options)
    converged = g_converged || stopped
    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    options.show_trace && print_header(method)
    _time = time()
    trace!(tr, d, state, iteration, method, options, _time - t0)
    ls_success::Bool = true
    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        # Convention: When `update_state!` is called, then `state` satisfies:
        # - `state.x`: Current state
        # - `state.f`: Objective function value of the current state, ie. `d(state.x)`
        # - `state.g_x` (if available): Gradient of the objective function at the current state, i.e. `gradient(d, state.x)`
        # - `state.H_x` (if available): Hessian of the objective function at the current state, i.e. `hessian(d, state.x)` 
        ls_success = !update_state!(d, state, method)
        if !ls_success
            break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        end

        # Update function value, gradient and Hessian matrix (skipped by some methods that already update those in `update_state!`)
        # TODO: Already perform in `update_state!`?
        update_fgh!(d, state, method)

        # Check convergence
        x_converged, f_converged, g_converged, f_increased =
            assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol + 1 : 0
        converged = x_converged || g_converged || (counter_f_tol > options.successive_f_tol)

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
                small_trustregion_radius = true
                stopped = true
            end
        end

        if hasproperty(state, :g_x) && !all(isfinite, state.g_x)
            options.show_warnings && @warn "Terminated early due to NaN in gradient."
            break
        end
        if hasproperty(state, :H_x) && !all(isfinite, state.H_x)
            options.show_warnings && @warn "Terminated early due to NaN in Hessian."
            break
        end
    end # while

    after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    Tf = typeof(state.f_x)
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
        small_trustregion_radius,
    )

    termination_code =
        _termination_code(d, g_residual(state), state, stopped_by, options)

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
        pick_best_f(f_incr_pick, state),
        iteration,
        Tf(options.x_abstol),
        Tf(options.x_reltol),
        x_abschange(state),
        x_relchange(state),
        Tf(options.f_abstol),
        Tf(options.f_reltol),
        f_abschange(state),
        f_relchange(state),
        Tf(options.g_abstol),
        g_residual(state),
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
    elseif (iszero(options.f_abstol) && f_abschange(state) <= options.f_abstol) ||
           (iszero(options.f_reltol) && f_relchange(state) <= options.f_reltol)
        TerminationCode.NoObjectiveChange
    elseif x_abschange(state) <= options.x_abstol || x_relchange(state) <= options.x_reltol
        TerminationCode.SmallXChange
    elseif f_abschange(state) <= options.f_abstol ||
           f_relchange(state) <= options.f_reltol
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
    elseif !isfinite(state.f_x)
        TerminationCode.ObjectiveNotFinite
    elseif hasproperty(state, :g_x) && !all(isfinite, state.g_x)
        TerminationCode.GradientNotFinite
    elseif hasproperty(state, :H_x) && !all(isfinite, state.H_x)
        TerminationCode.HessianNotFinite
    elseif stopped_by.small_trustregion_radius
        TerminationCode.SmallTrustRegionRadius
    else
        TerminationCode.NotImplemented
    end
end
