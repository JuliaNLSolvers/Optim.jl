"""
    AdaMax(; alpha=0.002, beta_mean=0.9, beta_var=0.999,
        epsilon=sqrt(eps(Float64)))

AdaMax is a first-order optimizer that uses an exponentially weighted first
gradient moment and an infinity-norm second-moment estimate. It is a variant
of Adam that can be useful when the objective or gradient is stochastic.
AdaMax does not perform a line search, so `alpha` may need to be tuned.

# Keyword Arguments

- `alpha`: Step-size parameter or callable scheduler. A scheduler is called as
  `alpha(iteration)`.
- `beta_mean`: Exponential decay factor for the first gradient moment.
- `beta_var`: Exponential decay factor for the second-moment estimate.
- `epsilon`: Positive numerical stabilizer. The default is
  `sqrt(eps(Float64))`.

# Fields

- `α`: Step-size parameter or scheduler stored by the optimizer.
- `β₁`: First-moment decay factor.
- `β₂`: Second-moment decay factor.
- `ϵ`: Numerical stabilizer.
- `manifold`: Manifold used for iterate and tangent-space operations. The
  keyword constructor uses [`Flat`](@ref).

# Returns

An initialized `AdaMax` optimizer that can be passed to [`optimize`](@ref).

# Examples

```julia
using Optim

f(x) = sum(abs2, x)
g!(G, x) = (G .= 2 .* x)
result = optimize(f, g!, [1.0, -1.0], AdaMax(alpha=0.002))
Optim.minimizer(result)
```

# References

- "Adam: A Method for Stochastic Optimization" (2014).
"""
struct AdaMax{Tα,T,Tm} <: FirstOrderOptimizer
    α::Tα
    β₁::T
    β₂::T
    ϵ::T
    manifold::Tm
end
AdaMax(; alpha = 0.002, beta_mean = 0.9, beta_var = 0.999, epsilon = sqrt(eps(Float64))) =
    AdaMax(alpha, beta_mean, beta_var, epsilon, Flat())
Base.summary(io::IO, ::AdaMax) = print(io, "AdaMax")
function default_options(method::AdaMax)
    (; allow_f_increases = true, iterations = 10_000)
end

mutable struct AdaMaxState{Tx,T,Tg} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    f_x::T
    x_previous::Tx
    f_x_previous::T
    s::Tx
    m::Tg
    u::Tg
    alpha::T
    iter::Int
end

function reset!(method::AdaMax, state::AdaMaxState, obj, x)
    # Update function value and gradient
    copyto!(state.x, x)
    retract!(method.manifold, state.x)
    f_x, g_x = NLSolversBase.value_gradient!(obj, state.x)
    copyto!(state.g_x, g_x)
    project_tangent!(method.manifold, state.g_x, state.x)
    state.f_x = f_x

    # Delete history
    fill!(state.x_previous, NaN)
    state.f_x_previous = oftype(state.f_x_previous, NaN)
    fill!(state.s, NaN)

    # Update momentum
    copyto!(state.m, state.g_x)
    fill!(state.u, false)

    return nothing
end

function _init_alpha(method::AdaMax)
    (; α) = method
    return α isa Real ? α : α(1)
end

function initial_state(method::AdaMax, options::Options, d, x0::AbstractArray{T}) where {T}
    # Compute function value and gradient
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = NLSolversBase.value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    AdaMaxState(
        x0, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        copy(g_x), # m
        zero(g_x), # u
        _init_alpha(method), # alpha
        0, # iter
    )
end

function update_state!(d, state::AdaMaxState, method::AdaMax)
    state.iter += 1

    # Update step size alpha if it is not constant
    if !(method.α isa Real)
        state.alpha = method.α(state.iter)
    end

    # Unpack parameters
    α = state.alpha
    (; β₁, β₂, ϵ) = method
    a = 1 - β₁

    (; g_x, m, u) = state
    m .= β₁ .* m .+ a .* g_x
    u .= max.(ϵ, max.(β₂ .* u, abs.(g_x))) # I know it's not there in the paper but if m and u start at 0 for some element... NaN occurs next

    # Update current state
    copyto!(state.x_previous, state.x)
    state.f_x_previous = state.f_x
    @. state.x = state.x - (α / (1 - β₁^state.iter)) * m / u

    false # no error
end

function trace!(tr, d, state::AdaMaxState, iteration::Integer, method::AdaMax, options::Options, curr_time = time())
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end
