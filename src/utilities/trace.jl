function trace!(tr, state, iteration, method::NelderMead, options)
    dt = Dict()
    if options.extended_trace
        dt["centroid"] = state.x_centroid
        dt["step_type"] = state.step_type
    end
    update!(tr,
    iteration,
    state.f_lowest,
    state.f_x,
    dt,
    options.store_trace,
    options.show_trace,
    options.show_every,
    options.callback)
end


function trace!(tr, state, iteration, method::Union{ParticleSwarm, SimulatedAnnealing}, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
    end
    g_norm = NaN
    update!(tr,
            state.iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function trace!(tr, state, iteration, method::BFGS, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["~inv(H)"] = copy(state.invH)
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(state.g, Inf)
    update!(tr,
    iteration,
    state.f_x,
    g_norm,
    dt,
    options.store_trace,
    options.show_trace,
    options.show_every,
    options.callback)
end

function trace!(tr, state, iteration, method::Union{LBFGS, AcceleratedGradientDescent, GradientDescent, MomentumGradientDescent, ConjugateGradient}, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(state.g, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function trace!(tr, state, iteration, method::Newton, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["h(x)"] = copy(state.H)
    end
    g_norm = vecnorm(state.g, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function trace!(tr, state, iteration, method::NewtonTrustRegion, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["h(x)"] = copy(state.H)
        dt["delta"] = copy(state.delta)
        dt["interior"] = state.interior
        dt["hard case"] = state.hard_case
        dt["reached_subproblem_solution"] = state.reached_subproblem_solution
        dt["lambda"] = state.lambda
    end
    g_norm = norm(state.g, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function trace!(tr, state, iteration, method::IPOptimizer, options)
    dt = Dict()
    dt["Lagrangian"] = state.L
    dt["μ"] = state.μ
    dt["ev"] = abs(state.ev)
    if options.extended_trace
        dt["α"] = state.alpha
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["h(x)"] = copy(state.H)
        if !isempty(state.bstate)
            dt["gtilde(x)"] = copy(state.gtilde)
            dt["bstate"] = copy(state.bstate)
            dt["bgrad"] = copy(state.bgrad)
            dt["c"] = copy(state.constr_c)
        end
    end
    g_norm = vecnorm(state.g, Inf) + vecnorm(state.bgrad, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
