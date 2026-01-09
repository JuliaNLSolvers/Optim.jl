function trace!(
    tr,
    d,
    state,
    iteration,
    method::Union{ZerothOrderOptimizer,SAMIN},
    options::Options,
    curr_time = time(),
)
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
    end
    update!(
        tr,
        state.iteration,
        state.f_x,
        NaN,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end

function assess_convergence(state::ZerothOrderState, d, options::Options)
    false, false, false, false
end

f_abschange(state::ZerothOrderState) = oftype(state.f_x, NaN)
f_relchange(state::ZerothOrderState) = oftype(state.f_x, NaN)
x_abschange(state::ZerothOrderState) = convert(real(eltype(state.x)), NaN)
x_relchange(state::ZerothOrderState) = convert(real(eltype(state.x)), NaN)
