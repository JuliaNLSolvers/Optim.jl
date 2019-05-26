log_temperature(t) = 1 / log(t)

constant_temperature(t) = 1.0

function default_neighbor!(x::AbstractArray{T}, x_proposal::AbstractArray) where T
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + T(randn()) # workaround because all types might not have randn
    end
    return
end

struct SimulatedAnnealing{Tn, Ttemp} <: ZerothOrderOptimizer
    neighbor!::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
end

"""
# SimulatedAnnealing
## Constructor
```julia
SimulatedAnnealing(; neighbor = default_neighbor!,
                     temperature = log_temperature,
                     keep_best::Bool = true)
```

The constructor takes 3 keywords:
* `neighbor = a!(x_proposed, x_current)`, a mutating function of the current `x`,
and the proposed `x`
* `T = b(iteration)`, a function of the current iteration that returns a temperature
* `p = c(f_proposal, f_current, T)`, a function of the current temperature, current
function value and proposed function value that returns an acceptance probability

## Description
Simulated Annealing is a derivative free method for optimization. It is based on the
Metropolis-Hastings algorithm that was originally used to generate samples from a
thermodynamics system, and is often used to generate draws from a posterior when doing
Bayesian inference. As such, it is a probabilistic method for finding the minimum of a
function, often over a quite large domains. For the historical reasons given above, the
algorithm uses terms such as cooling, temperature, and acceptance probabilities.
"""
SimulatedAnnealing(;neighbor = default_neighbor!,
                    temperature = log_temperature,
                    keep_best::Bool = true) =
  SimulatedAnnealing(neighbor, temperature, keep_best)

Base.summary(::SimulatedAnnealing) = "Simulated Annealing"

mutable struct SimulatedAnnealingState{Tx,T} <: ZerothOrderState
    x::Tx
    iteration::Int
    x_current::Tx
    x_proposal::Tx
    f_x_current::T
    f_proposal::T
end
# We don't have an f_x_previous in SimulatedAnnealing, so we need to special case these
pick_best_x(f_increased, state::SimulatedAnnealingState) = state.x
pick_best_f(f_increased, state::SimulatedAnnealingState, d) = value(d)

function initial_state(method::SimulatedAnnealing, options, d, initial_x::AbstractArray{T}) where T

    value!!(d, initial_x)

    # Store the best state ever visited
    best_x = copy(initial_x)
    SimulatedAnnealingState(copy(best_x), 1, best_x, copy(initial_x), value(d), value(d))
end

function update_state!(nd, state::SimulatedAnnealingState{Tx, T}, method::SimulatedAnnealing) where {Tx, T}

    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor!(state.x_current, state.x_proposal)

    # Evaluate the cost function at the proposed state
    state.f_proposal = value(nd, state.x_proposal)

    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copyto!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < value(nd)
            nd.F = state.f_proposal
            copyto!(state.x, state.x_proposal)
        end
    else
        # If proposal is inferior, we move to it with probability p
        p = exp(-(state.f_proposal - state.f_x_current) / t)
        if rand() <= p
            copyto!(state.x_current, state.x_proposal)
            state.f_x_current = state.f_proposal
        end
    end

    state.iteration += 1
    false
end
