"""
    Adam(; alpha=0.0001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8)

Adam is a first-order optimizer that updates the iterate using exponentially
weighted estimates of the first and second moments of the gradient. The
`alpha` keyword may be a number or a callable scheduler receiving the current
iteration number. Adam does not perform a line search, so `alpha` may need to
be tuned for the objective.

# Keyword Arguments

- `alpha`: Step-size parameter or callable scheduler. A scheduler is called as
  `alpha(iteration)`.
- `beta_mean`: Exponential decay factor for the first gradient moment. Values
  closer to one retain more history.
- `beta_var`: Exponential decay factor for the second gradient moment. Values
  closer to one retain more history.
- `epsilon`: Positive numerical stabilizer used in the denominator.

# Fields

- `α`: Step-size parameter or scheduler stored by the optimizer.
- `β₁`: First-moment decay factor.
- `β₂`: Second-moment decay factor.
- `ϵ`: Numerical stabilizer.
- `manifold`: Manifold used for iterate and tangent-space operations. The
  keyword constructor uses [`Flat`](@ref).

# Returns

An initialized `Adam` optimizer that can be passed to [`optimize`](@ref).

# Examples

```julia
using Optim

f(x) = sum(abs2, x)
g!(G, x) = (G .= 2 .* x)
result = optimize(f, g!, [1.0, -1.0], Adam(alpha=0.001))
Optim.minimizer(result)
```

To schedule the step size, pass a callable such as
`alpha = iteration -> 0.001 / sqrt(iteration)`.

# References

- "Adam: A Method for Stochastic Optimization" (2014).
"""
struct Adam{Tα,T,Tm} <: FirstOrderOptimizer
    α::Tα
    β₁::T
    β₂::T
    ϵ::T
    manifold::Tm
end
# could use epsilon = T->sqrt(eps(T)) and input the promoted type
Adam(; alpha = 0.0001, beta_mean = 0.9, beta_var = 0.999, epsilon = 1e-8) =
    Adam(alpha, beta_mean, beta_var, epsilon, Flat())
Base.summary(io::IO, ::Adam) = print(io, "Adam")
function default_options(method::Adam)
    (; allow_f_increases = true, iterations = 10_000)
end

mutable struct AdamState{Tx,T,Tg,Tu,Ti} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    f_x::T
    x_previous::Tx
    f_x_previous::T
    s::Tx
    m::Tg
    u::Tu
    alpha::T
    iter::Ti
end

function reset!(method::Adam, state::AdamState, obj, x)
    # Update function value and gradient
    copyto!(state.x, x)
    retract!(method.manifold, state.x)
    f_x, g_x = NLSolversBase.value_gradient!(obj, state.x)
    copyto!(state.g_x, g_x)
    project_tangent!(method.manifold, state.g_x, state.x)
    state.f_x = f_x

    # Reset history
    fill!(state.x_previous, NaN)
    state.f_x_previous = oftype(state.f_x_previous, NaN)
    fill!(state.s, NaN)

    # Reset momentum
    copyto!(state.m, state.g_x)
    fill!(state.u, false)

    return nothing
end

function _init_alpha(method::Adam)
    (; α) = method
    return α isa Real ? α : α(1)
end

function initial_state(method::Adam, ::Options, d, x0::AbstractArray)
    # Compute function value and gradient
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = NLSolversBase.value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    AdamState(
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

function update_state!(d, state::AdamState, method::Adam)
    state.iter += 1

    # Update α parameter if it is not constant
    if !(method.α isa Real)
        state.alpha = method.α(state.iter)
    end

    # Unpack parameters
    α = state.alpha
    (; β₁, β₂, ϵ) = method
    a = 1 - β₁
    b = 1 - β₂

    m, u = state.m, state.u
    v = u
    m .= β₁ .* m .+ a .* state.g_x
    v .= β₂ .* v .+ b .* state.g_x .^ 2
    #  m̂ = m./(1-β₁^state.iter)
    # v̂ = v./(1-β₂^state.iter)
    #@. z = z - α*m̂/(sqrt(v̂+ϵ))
    αₜ = α * sqrt(1 - β₂^state.iter) / (1 - β₁^state.iter)

    # Update current state
    copyto!(state.x_previous, state.x)
    state.f_x_previous = state.f_x
    @. state.x = state.x - αₜ * m / (sqrt(v) + ϵ)

    false # no error
end

function trace!(tr, d, state::AdamState, iteration::Integer, method::Adam, options::Options, curr_time = time())
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end
