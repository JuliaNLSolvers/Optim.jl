# First order methods trace, used by AcceleratedGradientDescent,
# ConjugateGradient, GradientDescent, LBFGS and MomentumGradientDescent
function common_trace!(
    tr,
    d,
    state,
    iteration::Integer,
    ::FirstOrderOptimizer,
    options::Options,
    curr_time = time(),
)
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["Current step size"] = state.alpha
    end
    g_norm = Base.maximum(abs, state.g_x) # Base.maximum !== maximum
    update!(
        tr,
        iteration,
        state.f_x,
        g_norm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
