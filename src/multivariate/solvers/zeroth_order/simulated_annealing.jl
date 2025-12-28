log_temperature(t) = 1 / log(t)

constant_temperature(t) = 1.0

function default_neighbor!(x::AbstractArray{T}, x_proposal::AbstractArray) where {T}
    @assert size(x) == size(x_proposal)
    for i = 1:length(x)
        @inbounds x_proposal[i] = x[i] + T(randn()) # workaround because all types might not have randn
    end
    return
end

struct SimulatedAnnealing{Tn,Ttemp} <: ZerothOrderOptimizer
    neighbor!::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
end

"""
# SimulatedAnnealing
## Constructor
```julia
SimulatedAnnealing(; neighbor = default_neighbor!,
                     temperature = log_temperature)
```

The constructor takes two keywords:
* `neighbor = a!(x_current, x_proposed)`, a mutating function of the current `x`,
and the proposed `x`
* `temperature = b(iteration)`, a function of the current iteration that returns a temperature

## Description
Simulated Annealing is a derivative free method for optimization. It is based on the
Metropolis-Hastings algorithm that was originally used to generate samples from a
thermodynamics system, and is often used to generate draws from a posterior when doing
Bayesian inference. As such, it is a probabilistic method for finding the minimum of a
function, often over a quite large domains. For the historical reasons given above, the
algorithm uses terms such as cooling, temperature, and acceptance probabilities.
"""
SimulatedAnnealing(;
    neighbor = default_neighbor!,
    temperature = log_temperature,
    keep_best::Bool = true,
) = SimulatedAnnealing(neighbor, temperature, keep_best)

Base.summary(io::IO, ::SimulatedAnnealing) = print(io, "Simulated Annealing")

mutable struct SimulatedAnnealingState{Tx,T} <: ZerothOrderState
    x::Tx            # Best state ever visited
    f_x::T           # Function value of the best state ever visited
    iteration::Int   # Iteration number
    x_current::Tx    # Current state
    f_x_current::T   # Function value of current state
    x_proposal::Tx   # Proposed state
    f_proposal::T    # Function value of proposed state
end

# We don't have an x_previous and f_x_previous in SimulatedAnnealing, so we need to special case these
# The best state ever visited is stored in x, and its function value in f_x
pick_best_x(::Bool, state::SimulatedAnnealingState) = state.x
pick_best_f(::Bool, state::SimulatedAnnealingState) = state.f_x

function initial_state(
    ::SimulatedAnnealing,
    ::Options,
    d,
    initial_x::AbstractArray,
)
    # Compute function value
    f_x = value!(d, initial_x)

    return SimulatedAnnealingState(
        copy(initial_x), # best state ever visited
        f_x, # function value of the best state ever visited
        1, # iteration
        copy(initial_x), # current state
        f_x, # function value of the current state
        fill!(similar(initial_x), NaN), # proposed state
        oftype(f_x, NaN), # function value of the proposed state
    )
end

function update_state!(
    nd,
    state::SimulatedAnnealingState{Tx,T},
    method::SimulatedAnnealing,
) where {Tx,T}
    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor!(state.x_current, state.x_proposal)

    # Evaluate the cost function at the proposed state
    state.f_proposal = value!(nd, state.x_proposal)

    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copyto!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < state.f_x
            copyto!(state.x, state.x_proposal)
            state.f_x = state.f_proposal
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
