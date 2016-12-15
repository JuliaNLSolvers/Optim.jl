typealias FirstOrderSolver Union{AcceleratedGradientDescent, ConjugateGradient, GradientDescent,
                                 MomentumGradientDescent, BFGS, LBFGS}
typealias SecondOrderSolver Union{Newton, NewtonTrustRegion}


function optimize(d,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    optimize(d, initial_x, method, options)
end
optimize(d, initial_x, options::OptimizationOptions) = optimize(d, initial_x, NelderMead(), options)

function optimize{F<:Function, G<:Function}(f::F,
                  g!::G,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    d = DifferentiableFunction(f, g!)
    optimize(d, initial_x, method, options)
end
function optimize{F<:Function, G<:Function}(f::F,
                  g!::G,
                  initial_x::Array,
                  options::OptimizationOptions)
    d = DifferentiableFunction(f, g!)
    optimize(d, initial_x, BFGS(), options)
end

function optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    d = TwiceDifferentiableFunction(f, g!, h!)
    optimize(d, initial_x, method, options)
end
function optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array,
                  options::OptimizationOptions = OptimizationOptions())
    d = TwiceDifferentiableFunction(f, g!, h!)
    optimize(d, initial_x, Newton(), options)
end

function optimize{F<:Function, T, M <: Union{FirstOrderSolver, SecondOrderSolver}}(f::F,
                  initial_x::Array{T},
                  method::M,
                  options::OptimizationOptions)
    if !options.autodiff
        if M <: FirstOrderSolver
            d = DifferentiableFunction(f)
        else
            error("No gradient or Hessian was provided. Either provide a gradient and Hessian, set autodiff = true in the OptimizationOptions if applicable, or choose a solver that doesn't require a Hessian.")
        end
    else
        gcfg = ForwardDiff.GradientConfig(initial_x)
        g! = (x, out) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        if M <: FirstOrderSolver
            d = DifferentiableFunction(f, g!, fg!)
        else
            hcfg = ForwardDiff.HessianConfig(initial_x)
            h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
            d = TwiceDifferentiableFunction(f, g!, fg!, h!)
        end
    end

    optimize(d, initial_x, method, options)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array,
                  method::Newton,
                  options::OptimizationOptions)
    if !options.autodiff
        error("No Hessian was provided. Either provide a Hessian, set autodiff = true in the OptimizationOptions if applicable, or choose a solver that doesn't require a Hessian.")
    else
        hcfg = ForwardDiff.HessianConfig(initial_x)
        h! = (x, out) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    end
    optimize(TwiceDifferentiableFunction(d.f, d.g!, d.fg!, h!), initial_x, method, options)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array,
                  method::NewtonTrustRegion,
                  options::OptimizationOptions)
    if !options.autodiff
        error("No Hessian was provided. Either provide a Hessian, set autodiff = true in the OptimizationOptions if applicable, or choose a solver that doesn't require a Hessian.")
    else
        hcfg = ForwardDiff.HessianConfig(initial_x)
        h! = (x, out) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    end
    optimize(TwiceDifferentiableFunction(d.f, d.g!, d.fg!, h!), initial_x, method, options)
end

update_g!(d, state, method) = nothing

function update_g!{M<:Union{FirstOrderSolver, Newton}}(d, state, method::M)
    # Update the function value and gradient
    state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1
end

update_h!(d, state, method) = nothing

# Update the Hessian
function update_h!(d, state, method::SecondOrderSolver)
    d.h!(state.x, state.H)
    state.h_calls += 1
end

after_while!(d, state, method, options) = nothing

function optimize{T, M<:Optimizer}(d, initial_x::Array{T}, method::M, options::OptimizationOptions)
    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end


    state = initial_state(method, options, d, initial_x)

    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false

    x_converged, f_converged = false, false
    g_converged = if typeof(method) <: NelderMead
        nmobjective(state.f_simplex, state.m, state.n) < options.g_tol
    elseif  typeof(method) <: ParticleSwarm || typeof(method) <: SimulatedAnnealing
        g_converged = false
    else
        vecnorm(state.g, Inf) < options.g_tol
    end

    converged = g_converged
    iteration = 0

    options.show_trace && print_header(method)
    trace!(tr, state, iteration, method, options)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        update_state!(d, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS)
        update_g!(d, state, method)
        x_converged, f_converged,
        g_converged, converged = assess_convergence(state, options)
        # We don't use the Hessian for anything if we have declared convergence,
        # so we might as well not make the (expensive) update if converged == true
        !converged && update_h!(d, state, method)

        # If tracing, update trace with trace!. If a callback is provided, it
        # should have boolean return value that controls the variable stopped_by_callback.
        # This allows for early stopping controlled by the callback.
        if tracing
            stopped_by_callback = trace!(tr, state, iteration, method, options)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false

        # Combine the two, so see if the stopped flag should be changed to true
        # and stop the while loop
        stopped = stopped_by_callback || stopped_by_time_limit ? true : false
    end # while

    after_while!(d, state, method, options)

    return MultivariateOptimizationResults(state.method_string,
                                            initial_x,
                                            state.x,
                                            Float64(state.f_x),
                                            iteration,
                                            iteration == options.iterations,
                                            x_converged,
                                            options.x_tol,
                                            f_converged,
                                            options.f_tol,
                                            g_converged,
                                            options.g_tol,
                                            tr,
                                            state.f_calls,
                                            state.g_calls,
                                            state.h_calls)
end

# Univariate OptimizationOptions
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
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    optimize(f, Float64(lower), Float64(upper), method;
             rel_tol = rel_tol,
             abs_tol = abs_tol,
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
