"""
# Adam
## Constant `alpha` case (default) constructor:

```julia
    Adam(; alpha=0.0001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8)
```

## Scheduled `alpha` case constructor:

Alternative to the above (default) usage where `alpha` is a fixed constant for
all the iterations, the following constructor provides flexibility for `alpha`
to be a callable object (a scheduler) that maps the current iteration count to
a value of `alpha` that is to-be used for the current optimization iteraion's
update step. This helps us in scheduling `alpha` over the iterations as
desired, using the following usage,

```julia
    # Let alpha_scheduler be iteration -> alpha value mapping callable object
    Adam(; alpha=alpha_scheduler, other_kwargs...)
```

## Description
Adam is a gradient based optimizer that choses its search direction by building
up estimates of the first two moments of the gradient vector. This makes it
suitable for problems with a stochastic objective and thus gradient. The method
is introduced in [1] where the related AdaMax method is also introduced, see
`?AdaMax` for more information on that method.

## References
[1] https://arxiv.org/abs/1412.6980
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

function initial_state(method::Adam, ::Options, d, initial_x::AbstractArray)
    # Compute function value and gradient
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    f_x, g_x = NLSolversBase.value_gradient!(d, initial_x)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, initial_x)

    AdamState(
        initial_x, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(initial_x), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(initial_x), NaN), # Maintain current search direction in state.s
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
