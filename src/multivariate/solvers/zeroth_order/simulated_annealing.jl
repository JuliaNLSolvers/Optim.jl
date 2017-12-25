log_temperature(t::Real) = 1 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Array, x_proposal::Array)
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + randn()
    end
    return
end

struct SimulatedAnnealing{Tn, Ttemp} <: ZerothOrderOptimizer
    neighbor!::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
end

SimulatedAnnealing(;neighbor = default_neighbor!,
                    temperature = log_temperature,
                    keep_best::Bool = true) =
  SimulatedAnnealing(neighbor, temperature, keep_best)

Base.summary(::SimulatedAnnealing) = "Simulated Annealing"

mutable struct SimulatedAnnealingState{T, N} <: ZerothOrderState
    x::Array{T,N}
    iteration::Int
    x_current::Array{T, N}
    x_proposal::Array{T, N}
    f_x_current::T
    f_proposal::T
end
# We don't have an f_x_previous in SimulatedAnnealing, so we need to special case these
pick_best_x(f_increased, state::SimulatedAnnealingState) = state.x
pick_best_f(f_increased, state::SimulatedAnnealingState, d) = value(d)

function initial_state(method::SimulatedAnnealing, options, d, initial_x::Array{T}) where T

    value!!(d, initial_x)

    # Store the best state ever visited
    best_x = copy(initial_x)
    SimulatedAnnealingState(copy(best_x), 1, best_x, similar(initial_x), value(d), value(d))
end

function update_state!(nd, state::SimulatedAnnealingState{T}, method::SimulatedAnnealing) where T

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
        if state.f_proposal < value(nd)
            nd.F = state.f_proposal
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
