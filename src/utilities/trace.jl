# First order methods trace, used by AcceleratedGradientDescent,
# ConjugateGradient, GradientDescent, LBFGS and MomentumGradientDescent
function common_trace!(tr, d, state, iteration, method::FirstOrderOptimizer, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = norm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
