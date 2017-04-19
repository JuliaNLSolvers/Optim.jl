const FirstOrderSolver = Union{AcceleratedGradientDescent, ConjugateGradient, GradientDescent,
                               MomentumGradientDescent, BFGS, LBFGS}
const SecondOrderSolver = Union{Newton, NewtonTrustRegion}
# Multivariate optimization
function check_kwargs(kwargs, fallback_method)
    kws = Dict{Symbol, Any}()
    method = nothing
    for kwarg in kwargs
        if kwarg[1] != :method
            kws[kwarg[1]] = kwarg[2]
        else
            method = kwarg[2]
        end
    end

    if method == nothing
        method = fallback_method
    end
    kws, method
end

function optimize{F<:Function}(f::F, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, NelderMead())
    optimize(f, initial_x, method, Options(; checked_kwargs...))
end

function optimize{F<:Function, G<:Function}(f::F, g!::G, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, BFGS())
    optimize(f, g!, initial_x, method, Options(;checked_kwargs...))
end
function optimize(d::OnceDifferentiable, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, BFGS())
    optimize(d, initial_x, method, Options(checked_kwargs...))
end

function optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, Newton())
    optimize(f, g!, h!, initial_x, method, Options(checked_kwargs...))
end

function optimize(d::TwiceDifferentiable, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, Newton())
    optimize(d, initial_x, method, Options(;kwargs...))
end

optimize(f::Function, initial_x::Array, options::Options) = optimize(NonDifferentiable(f, initial_x), initial_x, NelderMead(), options)
optimize(f::Function, initial_x::Array, method::Optimizer, options::Options = Options()) = optimize(NonDifferentiable(f, initial_x), initial_x, method, options)
optimize(d::OnceDifferentiable, initial_x::Array, options::Options) = optimize(d, initial_x, BFGS(), options)
optimize(d::TwiceDifferentiable, initial_x::Array, options::Options) = optimize(d, initial_x, Newton(), options)

function optimize{F<:Function, G<:Function}(f::F,
                  g!::G,
                  initial_x::Array,
                  method::Optimizer,
                  options::Options = Options())
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, method, options)
end
function optimize{F<:Function, G<:Function}(f::F,
                  g!::G,
                  initial_x::Array,
                  options::Options)
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, BFGS(), options)
end

function optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array,
                  method::Optimizer,
                  options::Options = Options())
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, method, options)
end
function optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array,
                  options)
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, Newton(), options)
end

function optimize{F<:Function, T, M <: Union{FirstOrderSolver, SecondOrderSolver}}(f::F,
                  initial_x::Array{T},
                  method::M,
                  options::Options)
    if M <: FirstOrderSolver
        d = OnceDifferentiable(f, initial_x)
    else
        d = TwiceDifferentiable(f, initial_x)
    end
    optimize(d, initial_x, method, options)
end

function optimize(d::OnceDifferentiable,
                  initial_x::Array,
                  method::SecondOrderSolver,
                  options::Options)
    optimize(TwiceDifferentiable(d), initial_x, method, options)
end

update_g!(d, state, method) = nothing
function update_g!{M<:Union{FirstOrderSolver, Newton}}(d, state, method::M)
    # Update the function value and gradient
    value_gradient!(d, state.x)
end

# Update the Hessian
update_h!(d, state, method) = nothing
update_h!(d, state, method::SecondOrderSolver) = hessian!(d, state.x)

after_while!(d, state, method, options) = nothing

function optimize{D<:AbstractObjective, T, M<:Optimizer}(d::D, initial_x::Array{T}, method::M,
    options::Options = Options(), state = initial_state(method, options, d, initial_x))

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end

    n = length(initial_x)
    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false
    g_converged = if typeof(method) <: NelderMead
        nmobjective(state.f_simplex, state.m, n) < options.g_tol
    elseif  typeof(method) <: ParticleSwarm || typeof(method) <: SimulatedAnnealing
        g_converged = false
    else
        vecnorm(gradient(d), Inf) < options.g_tol
    end

    converged = g_converged
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

    return MultivariateOptimizationResults(Optim.method(method),
                                            initial_x,
                                            f_increased ? state.x_previous : state.x,
                                            f_increased ? state.f_x_previous : value(d),
                                            iteration,
                                            iteration == options.iterations,
                                            x_converged,
                                            options.x_tol,
                                            f_converged,
                                            options.f_tol,
                                            g_converged,
                                            options.g_tol,
                                            f_increased,
                                            tr,
                                            f_calls(d),
                                            g_calls(d),
                                            h_calls(d))
end

# Univariate Options
function optimize{F<:Function, T <: AbstractFloat}(f::F,
                                      lower::T,
                                      upper::T;
                                      method = Brent(),
                                      rel_tol::Real = sqrt(eps(T)),
                                      abs_tol::Real = eps(T),
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      callback = nothing,
                                      show_every = 1,
                                      extended_trace::Bool = false)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end

    show_trace && print_header(method)

    optimize(f, lower, upper, method;
             rel_tol = T(rel_tol),
             abs_tol = T(abs_tol),
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             show_every = show_every,
             callback = callback,
             extended_trace = extended_trace)
end

function optimize{F<:Function}(f::F,
                  lower::Real,
                  upper::Real;
                  kwargs...)
    optimize(f,
             Float64(lower),
             Float64(upper);
             kwargs...)
end

function optimize{F<:Function}(f::F,
                  lower::Real,
                  upper::Real,
                  mo::Union{Brent, GoldenSection};
                  kwargs...)
    optimize(f,
             Float64(lower),
             Float64(upper),
             mo;
             kwargs...)
end
