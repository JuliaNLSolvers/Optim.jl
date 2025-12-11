"""
# AdaMax
## Constant `alpha` case (default) constructor:

```julia
    AdaMax(; alpha=0.002, beta_mean=0.9, beta_var=0.999, epsilon=1e-8)
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
    AdaMax(; alpha=alpha_scheduler, other_kwargs...)
```

## Description
AdaMax is a gradient based optimizer that choses its search direction by
building up estimates of the first two moments of the gradient vector. This
makes it suitable for problems with a stochastic objective and thus gradient.
The method is introduced in [1] where the related Adam method is also
introduced, see `?Adam` for more information on that method.

## References
[1] https://arxiv.org/abs/1412.6980
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

function initial_state(method::AdaMax, options::Options, d, initial_x::AbstractArray{T}) where {T}
    # Compute function value and gradient
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    f_x, g_x = NLSolversBase.value_gradient!(d, initial_x)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, initial_x)

    AdaMaxState(
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
