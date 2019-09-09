update_g!(d, state, method) = nothing
function update_g!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton}
    # Update the function value and gradient
    value_gradient!(d, state.x)
    if M <: FirstOrderOptimizer #only for methods that support manifold optimization
        project_tangent!(method.manifold, gradient(d), state.x)
    end
end
update_fg!(d, state, method) = nothing
update_fg!(d, state, method::ZerothOrderOptimizer) = value!(d, state.x)
function update_fg!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton}
    value_gradient!(d, state.x)
    if M <: FirstOrderOptimizer #only for methods that support manifold optimization
        project_tangent!(method.manifold, gradient(d), state.x)
    end
end

# Update the Hessian
update_h!(d, state, method) = nothing
update_h!(d, state, method::SecondOrderOptimizer) = hessian!(d, state.x)

after_while!(d, state, method, options) = nothing

function initial_convergence(d, state, method::AbstractOptimizer, initial_x, options)
    gradient!(d, initial_x)
    norm(gradient(d), Inf) < options.g_abstol
end
initial_convergence(d, state, method::ZerothOrderOptimizer, initial_x, options) = false

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

@with_kw struct IteratorState{IT <: OptimIterator, TR <: OptimizationTrace}
    # Put `OptimIterator` in iterator state so that `OptimizationResults` can
    # be constructed from `IteratorState`.
    iter::IT

    t0::Float64
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
    @unpack d, initial_x, method, options, state = iter
    if istate === nothing
        t0 = time() # Initial time stamp used to control early stopping by options.time_limit

        tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
        tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
        stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
        f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
        x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

        g_converged = initial_convergence(d, state, method, initial_x, options)
        converged = g_converged

        # prepare iteration counter (used to make "initial state" trace entry)
        iteration = 0

        options.show_trace && print_header(method)
        trace!(tr, d, state, iteration, method, options, time()-t0)
        ls_success::Bool = true

        # Note: `optimize` depends on that first iteration always yields something
        # (i.e., `iterate` does _not_ return a `nothing` when `istate === nothing`).
    else
        @unpack_IteratorState istate

        !converged && !stopped && iteration < options.iterations || return nothing

        iteration += 1

        ls_failed = update_state!(d, state, method)
        if !ls_success
            # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
            return nothing
        end
        update_g!(d, state, method) # TODO: Should this be `update_fg!`?

        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol+1 : 0
        converged = converged | (counter_f_tol > options.successive_f_tol)

        !converged && update_h!(d, state, method) # only relevant if not converged

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options, time()-t0)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > options.time_limit
        f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            stopped = true
        end
    end

    new_istate = IteratorState(
        iter,
        t0,
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
    @unpack_IteratorState istate
    @unpack d, initial_x, method, options, state = iter

    after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    Tf = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases

    T = typeof(options.x_abstol)
    Tx = typeof(initial_x)

    return MultivariateOptimizationResults{typeof(method),T,Tx,typeof(x_abschange(state)),Tf,typeof(tr), Bool}(method,
                                        initial_x,
                                        pick_best_x(f_incr_pick, state),
                                        pick_best_f(f_incr_pick, state, d),
                                        iteration,
                                        iteration == options.iterations,
                                        x_converged,
                                        Tf(options.x_abstol),
                                        Tf(options.x_reltol),
                                        x_abschange(state),
                                        x_relchange(state),
                                        f_converged,
                                        Tf(options.f_reltol),
                                        Tf(options.f_abstol),
                                        f_abschange(d, state),
                                        f_relchange(d, state),
                                        g_converged,
                                        Tf(options.g_abstol),
                                        g_residual(d),
                                        f_increased,
                                        tr,
                                        f_calls(d),
                                        g_calls(d),
                                        h_calls(d),
                                        !ls_success)
end

function optimizing(d::D, initial_x::Tx, method::M,
                    options::Options = Options(;default_options(method)...),
                    state = initial_state(method, options, d, initial_x)) where {D<:AbstractObjective, M<:AbstractOptimizer, Tx <: AbstractArray}
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("You cannot use NelderMead for univariate problems. Alternatively, use either interval bound univariate optimization, or another method such as BFGS or Newton.")
    end
    return OptimIterator(d, initial_x, method, options, state)
end

# Derive `IteratorState` accessors from `MultivariateOptimizationResults` accessors.
for f in [
    :(Base.summary)
    :minimizer
    :minimum
    :iterations
    :iteration_limit_reached
    :trace
    :x_trace
    :f_trace
    :f_calls
    :converged
    :g_norm_trace
    :g_calls
    :x_converged
    :f_converged
    :g_converged
    :initial_state
]
    @eval $f(istate::IteratorState) = $f(OptimizationResults(istate))
end
