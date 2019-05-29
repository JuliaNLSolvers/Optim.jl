function trace!(tr, d, state, iteration, method::ZerothOrderOptimizer, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
    end
    update!(tr,
            state.iteration,
            d.F,
            NaN,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function assess_convergence(state::ZerothOrderState, d, options::Options)
    false, false, false, false, false
end

f_abschange(d::AbstractObjective, state::ZerothOrderState) = convert(typeof(value(d)), NaN)
f_relchange(d::AbstractObjective, state::ZerothOrderState) = convert(typeof(value(d)), NaN)
x_abschange(state::ZerothOrderState) = convert(real(eltype(state.x)), NaN)
x_relchange(state::ZerothOrderState) = convert(real(eltype(state.x)), NaN)
