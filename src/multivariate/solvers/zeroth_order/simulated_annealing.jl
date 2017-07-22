log_temperature(t::Real) = 1 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Array, x_proposal::Array)
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + randn()
    end
    return
end

struct SimulatedAnnealing{Tn, Ttemp} <: Optimizer
    neighbor!::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
end

SimulatedAnnealing(;neighbor = default_neighbor!,
                    temperature = log_temperature,
                    keep_best::Bool = true) =
  SimulatedAnnealing(neighbor, temperature, keep_best)

Base.summary(::SimulatedAnnealing) = "Simulated Annealing"

mutable struct SimulatedAnnealingState{T, N}
    x::Array{T,N}
    iteration::Int
    x_current::Array{T, N}
    x_proposal::Array{T, N}
    f_x_current::T
    f_proposal::T
end

function initial_state{T}(method::SimulatedAnnealing, options, f, initial_x::Array{T})
    # Count number of parameters
    n = length(initial_x)
    value!(f, initial_x)

    # Store the best state ever visited
    best_x = copy(initial_x)
    SimulatedAnnealingState(copy(best_x), 1, best_x, similar(initial_x), f.f_x, f.f_x)
end

function update_state!{T}(nd, state::SimulatedAnnealingState{T}, method::SimulatedAnnealing)

    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor!(state.x_current, state.x_proposal)

    # Evaluate the cost function at the proposed state
    state.f_proposal = value(nd, state.x_proposal)

    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copy!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < nd.f_x
            nd.f_x = state.f_proposal
            copy!(state.x, state.x_proposal)
        end
    else
        # If proposal is inferior, we move to it with probability p
        p = exp(-(state.f_proposal - state.f_x_current) / t)
        if rand() <= p
            copy!(state.x_current, state.x_proposal)
            state.f_x_current = state.f_proposal
        end
    end

    state.iteration += 1
    false
end

function assess_convergence(state::SimulatedAnnealingState, d, options)
    false, false, false, false, false
end

function trace!(tr, d, state, iteration, method::SimulatedAnnealing, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
    end
    update!(tr,
            state.iteration,
            d.f_x,
            NaN,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
