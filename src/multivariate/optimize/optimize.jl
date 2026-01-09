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
struct OptimIterator{
    D<:AbstractObjective,
    M<:AbstractOptimizer,
    Tx<:AbstractArray,
    O<:Options,
    S,
}
    d::D
    initial_x::Tx
    method::M
    options::O
    state::S
end
_method(r::OptimIterator) = r.method

Base.IteratorSize(::Type{<:OptimIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:OptimIterator}) = Base.HasEltype()
Base.eltype(::Type{<:OptimIterator}) = IteratorState

struct IteratorState{TR<:OptimizationTrace}
    t0::Float64
    _time::Float64
    tr::TR
    tracing::Bool
    stopped::Bool
    stopped_by_callback::Bool
    stopped_by_time_limit::Bool
    f_limit_reached::Bool
    g_limit_reached::Bool
    h_limit_reached::Bool
    x_converged::Bool
    f_converged::Bool
    f_increased::Bool
    counter_f_tol::Int
    g_converged::Bool
    converged::Bool
    iteration::Int
    ls_success::Bool
end

function Base.iterate(iter::OptimIterator, istate = nothing)
    (; d, initial_x, method, options, state) = iter
    if istate === nothing
        t0 = time() # Initial time stamp used to control early stopping by options.time_limit
        tr = OptimizationTrace{typeof(value(d)),typeof(method)}()
        tracing =
            options.store_trace ||
            options.show_trace ||
            options.extended_trace ||
            options.callback != nothing
        stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
        f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
        x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0
        small_trustregion_radius = false

        g_converged, stopped = initial_convergence(d, state, method, initial_x, options)
        converged = g_converged

        # prepare iteration counter (used to make "initial state" trace entry)
        iteration = 0

        options.show_trace && print_header(method)
        _time = time()
        trace!(tr, d, state, iteration, method, options, _time - t0)
        ls_success::Bool = true
        stopped_by_callback = callback !== nothing && callback(state)
        stopped |= stopped_by_callback
    
        # Note: `optimize` depends on that first iteration always yields something
        # (i.e., `iterate` does _not_ return a `nothing` when `istate === nothing`).
    else
        (;
            t0,
            _time,
            tr,
            tracing,
            stopped,
            stopped_by_callback,
            stopped_by_time_limit,
            f_limit_reached,
            g_limit_reached,
            h_limit_reached,
            x_converged,
            f_converged,
            f_increased,
            counter_f_tol,
            g_converged,
            converged,
            iteration,
            ls_success,
        ) = istate

        !converged && !stopped && iteration < options.iterations || return nothing

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

        # update trace
        if tracing
            trace!(tr, d, state, iteration, method, options, time() - t0)
        end
        # callbacks can stop routine early by returning true
        if callback !== nothing
            stopped_by_callback = callback(state)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        _time = time()
        stopped_by_time_limit = _time - t0 > options.time_limit
        f_limit_reached =
            options.f_calls_limit > 0 && NLSolversBase.f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached =
            options.g_calls_limit > 0 && (NLSolversBase.g_calls(d) + NLSolversBase.jvp_calls(d)) >= options.g_calls_limit ? true : false
        h_limit_reached =
            options.h_calls_limit > 0 && (NLSolversBase.h_calls(d) + NLSolversBase.hvp_calls(d)) >= options.h_calls_limit ? true : false

        if method isa NewtonTrustRegion
            # If the trust region radius keeps on reducing we need to stop
            # because something is wrong. Wrong gradients or a non-differentiability
            # at the solution could be explanations.
            if state.delta ≤ method.delta_min
                stopped = true
            end
        end
        if (f_increased && !options.allow_f_increases) ||
           stopped_by_callback ||
           stopped_by_time_limit ||
           f_limit_reached ||
           g_limit_reached ||
           h_limit_reached
            stopped = true
        end
    end

    new_istate = IteratorState(
        t0,
        _time,
        tr,
        tracing,
        stopped,
        stopped_by_callback,
        stopped_by_time_limit,
        f_limit_reached,
        g_limit_reached,
        h_limit_reached,
        x_converged,
        f_converged,
        f_increased,
        counter_f_tol,
        g_converged,
        converged,
        iteration,
        ls_success,
    )
    return new_istate, new_istate
end

function OptimizationResults(iter::OptimIterator, istate::IteratorState)
    (;
        t0,
        _time,
        tr,
        tracing,
        stopped,
        stopped_by_callback,
        stopped_by_time_limit,
        f_limit_reached,
        g_limit_reached,
        h_limit_reached,
        x_converged,
        f_converged,
        f_increased,
        counter_f_tol,
        g_converged,
        converged,
        iteration,
        ls_success,
    ) = istate
    (; d, initial_x, method, options, state) = iter
        if method isa NewtonTrustRegion
            # If the trust region radius keeps on reducing we need to stop
            # because something is wrong. Wrong gradients or a non-differentiability
            # at the solution could be explanations.
            if state.delta ≤ method.delta_min
                small_trustregion_radius = true
                stopped = true
            end
        end

    if g_calls(d) > 0 && !all(isfinite, gradient(d))
        options.show_warnings && @warn "Terminated early due to NaN in gradient."
    end
    if h_calls(d) > 0 && !(d isa TwiceDifferentiableHV) && !all(isfinite, hessian(d))
        options.show_warnings && @warn "Terminated early due to NaN in Hessian."
    end

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    Tf = typeof(state.f_x)

    f_incr_pick = f_increased && !options.allow_f_increases
    stopped_by = (
        x_converged,
        f_converged,
        g_converged,
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
        NLSolversBase.f_calls(d),
        NLSolversBase.g_calls(d),
        NLSolversBase.jvp_calls(d),
        NLSolversBase.h_calls(d),
        NLSolversBase.hvp_calls(d),
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
    elseif stopped_by.ls_failed
        TerminationCode.FailedLinesearch
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

function optimizing(
    d::D,
    initial_x::Tx,
    method::M,
    options::Options = Options(; default_options(method)...),
    state = initial_state(method, options, d, initial_x),
) where {D<:AbstractObjective,M<:AbstractOptimizer,Tx<:AbstractArray}
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error(
            "You cannot use NelderMead for univariate problems. Alternatively, use either interval bound univariate optimization, or another method such as BFGS or Newton.",
        )
    end
    return OptimIterator(d, initial_x, method, options, state)
end

# we can just check minimum, as we've earlier enforced same types/eltypes
# in variables besides the option settings

function minimizer(iter::OptimIterator, istate::IteratorState)
    (; f_increased) = istate
    (; options, state) = iter
    f_incr_pick = f_increased && !options.allow_f_increases
    return pick_best_x(f_incr_pick, state)
end

function minimum(iter::OptimIterator, istate::IteratorState)
    (; f_increased) = istate
    (; d, options, state) = iter
    f_incr_pick = f_increased && !options.allow_f_increases
    return pick_best_f(f_incr_pick, state, d)
end

iterations(istate::IteratorState) = istate.iteration
iteration_limit_reached(iter::OptimIterator, istate::IteratorState) =
    istate.iteration == iter.options.iterations # this should be a precalculated one like the others
trace(istate::IteratorState) = istate.tr

converged(istate::IteratorState) = istate.converged
x_converged(istate::IteratorState) = istate.x_converged
f_converged(istate::IteratorState) = istate.f_converged
g_converged(istate::IteratorState) = istate.g_converged
initial_state(iter::OptimIterator) = iter.initial_x

f_calls(iter::OptimIterator) = f_calls(iter.d)
g_calls(iter::OptimIterator) = g_calls(iter.d)
h_calls(iter::OptimIterator) = h_calls(iter.d)
