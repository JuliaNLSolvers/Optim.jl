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

# Split zeroth-order complex into real and imaginary parts
function optimize(d::D, initial_x::Tx, method::M,
                  options::Options{T, TCallback} = nothing,
                  state = nothing) where {D<:NonDifferentiable, M<:AbstractOptimizer, Tx <: AbstractArray{S}, T, TCallback} where S <: Complex{R} where R
    initial_ri = similar(initial_x, R, 2, size(initial_x)...)
    selectdim(initial_ri, 1, 1) .= real(initial_x)
    selectdim(initial_ri, 1, 2) .= imag(initial_x)
    if isnothing(state)
        state = initial_state(method, options, d, initial_ri)
    end
    if isnothing(options)
        options = Options(;default_options(method)...)
    end
    result = optimize(d, initial_ri, method, options, state)
    args = []
    for s in fieldnames(typeof(result))
        arg = @eval result.$s
        if s ∈ [:minimizer, :initial_x]
            arg = complex.(eachslice(arg, dims=1)...)
        end
        push!(args, arg)
    end
    MultivariateOptimizationResults(args...)
end

function optimize(d::D, initial_x::Tx, method::M,
                  options::Options{T, TCallback} = Options(;default_options(method)...),
                  state = initial_state(method, options, d, initial_x)) where {D<:AbstractObjective, M<:AbstractOptimizer, Tx <: AbstractArray, T, TCallback}
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("You cannot use NelderMead for univariate problems. Alternatively, use either interval bound univariate optimization, or another method such as BFGS or Newton.")
    end

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
    _time = time()
    trace!(tr, d, state, iteration, method, options, _time-t0)
    ls_success::Bool = true
    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        ls_success = !update_state!(d, state, method)
        if !ls_success
            break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        end
        update_g!(d, state, method) # TODO: Should this be `update_fg!`?

        x_converged, f_converged,
        g_converged, f_increased = assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol+1 : 0
        converged = x_converged || g_converged || (counter_f_tol > options.successive_f_tol)

        if !(converged && method isa Newton)
            update_h!(d, state, method) # only relevant if not converged
        end
        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options, time()-t0)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        _time = time()
        stopped_by_time_limit = _time-t0 > options.time_limit
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
    Tf = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases
    stopped_by =(f_limit_reached=f_limit_reached,
                 g_limit_reached=g_limit_reached,
                 h_limit_reached=h_limit_reached,
                 time_limit=stopped_by_time_limit,
                 callback=stopped_by_callback,
                 f_increased=f_incr_pick)
    return MultivariateOptimizationResults{typeof(method),T,Tx,typeof(x_abschange(state)),Tf,typeof(tr), Bool, typeof(stopped_by)}(method,
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
                                        Tf(options.f_abstol),
                                        Tf(options.f_reltol),
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
                                        ls_success,
                                        options.time_limit,
                                        _time-t0,
                                        stopped_by,
                                        )
end
