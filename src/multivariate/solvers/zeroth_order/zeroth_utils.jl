const ZerothOrderStates = Union{NelderMeadState,ParticleSwarmState,SimulatedAnnealingState}

function trace!(tr, d, state, iteration, method::ZerothOrderOptimizer, options)
    dt = Dict()
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

function assess_convergence(state::ZerothOrderStates, d, options)
    false, false, false, false, false
end

f_abschange(d::AbstractObjective, state::ZerothOrderStates) = convert(typeof(value(d)), NaN)
x_abschange(state::ZerothOrderStates) = convert(eltype(state.x), NaN)
