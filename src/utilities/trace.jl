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
