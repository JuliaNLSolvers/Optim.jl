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
struct OptimIterator{D <: AbstractObjective, M <: AbstractOptimizer, Tx <: AbstractArray, O <: Options, S}
    d::D
    initial_x::Tx
    method::M
    options::O
    state::S
end

Base.IteratorSize(::Type{<:OptimIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:OptimIterator}) = Base.HasEltype()
Base.eltype(::Type{<:OptimIterator}) = IteratorState

struct IteratorState{IT <: OptimIterator, TR <: OptimizationTrace}
    # Put `OptimIterator` in iterator state so that `OptimizationResults` can
    # be constructed from `IteratorState`.
    iter::IT
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
    (;d, initial_x, method, options, state) = iter
    if istate === nothing
        t0 = time() # Initial time stamp used to control early stopping by options.time_limit
        tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
        tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
        stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
        f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
        x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

        g_converged, stopped = initial_convergence(d, state, method, initial_x, options)
        converged = g_converged

        # prepare iteration counter (used to make "initial state" trace entry)
        iteration = 0

        options.show_trace && print_header(method)
        _time = time()
        trace!(tr, d, state, iteration, method, options, _time-t0)
        ls_success::Bool = true

        # Note: `optimize` depends on that first iteration always yields something
        # (i.e., `iterate` does _not_ return a `nothing` when `istate === nothing`).
    else
        (;iter,
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
         ls_success) = istate

        !converged && !stopped && iteration < options.iterations || return nothing

        iteration += 1
        ls_success = !update_state!(d, state, method)
        if !ls_success
            # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
            return nothing
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
    end

    new_istate = IteratorState(
        iter,
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

function OptimizationResults(istate::IteratorState)
    (;iter,
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
          ls_success) = istate
    (;d, initial_x, method, options, state) = iter

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
        end
        if h_calls(d) > 0 && !(d isa TwiceDifferentiableHV) && !all(isfinite, hessian(d))
            options.show_warnings && @warn "Terminated early due to NaN in Hessian."
        end

    after_while!(d, state, method, options)

    Tf = typeof(value(d))
    Tx = typeof(state.x)
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

function optimizing(d::D, initial_x::Tx, method::M,
                    options::Options = Options(;default_options(method)...),
                    state = initial_state(method, options, d, initial_x)) where {D<:AbstractObjective, M<:AbstractOptimizer, Tx <: AbstractArray}
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("You cannot use NelderMead for univariate problems. Alternatively, use either interval bound univariate optimization, or another method such as BFGS or Newton.")
    end
    return OptimIterator(d, initial_x, method, options, state)
end

_method(istate::IteratorState) = istate.iter.method

# we can just check minimum, as we've earlier enforced same types/eltypes
# in variables besides the option settings

function minimizer(istate::IteratorState)
    (;iter, f_increased) = istate
    (;options, state) = iter
    f_incr_pick = f_increased && !options.allow_f_increases
    return pick_best_x(f_incr_pick, state)
end

function minimum(istate::IteratorState)
    (;iter, f_increased) = istate
    (;d, options, state) = iter
    f_incr_pick = f_increased && !options.allow_f_increases
    return pick_best_f(f_incr_pick, state, d)
end

iterations(istate::IteratorState) = istate.iteration
iteration_limit_reached(istate::IteratorState) = istate.iteration == istate.iter.options.iterations
trace(istate::IteratorState) = istate.tr

converged(istate::IteratorState) = istate.converged
x_converged(istate::IteratorState) = istate.x_converged
f_converged(istate::IteratorState) = istate.f_converged
g_converged(istate::IteratorState) = istate.g_converged
initial_state(istate::IteratorState) = istate.iter.initial_x

f_calls(istate::IteratorState) = f_calls(istate.iter.d)
g_calls(istate::IteratorState) = g_calls(istate.iter.d)
h_calls(istate::IteratorState) = h_calls(istate.iter.d)
