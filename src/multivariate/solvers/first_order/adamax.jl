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


mutable struct AdaMaxState{Tx,T,Tm,Tu,Ti} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    f_x_previous::T
    s::Tx
    m::Tm
    u::Tu
    alpha::T
    iter::Ti
end
function reset!(method, state::AdaMaxState, obj, x)
    value_gradient!!(obj, x)
end

function _get_init_params(method::AdaMax{T}) where {T<:Real}
    method.α, method.β₁, method.β₂
end

function _get_init_params(method::AdaMax)
    method.α(1), method.β₁, method.β₂
end

function initial_state(method::AdaMax, options, d, initial_x::AbstractArray{T}) where {T}
    initial_x = copy(initial_x)

    value_gradient!!(d, initial_x)
    α, β₁, β₂ = _get_init_params(method)

    m = copy(gradient(d))
    u = zero(m)
    iter = 0

    AdaMaxState(
        initial_x, # Maintain current state in state.x
        copy(initial_x), # Maintain previous state in state.x_previous
        real(T(NaN)), # Store previous f in state.f_x_previous
        similar(initial_x), # Maintain current search direction in state.s
        m,
        u,
        α,
        iter,
    )
end

function _update_iter_alpha_in_state!(state::AdaMaxState, method::AdaMax{T}) where {T<:Real}

    state.iter = state.iter + 1
end

function _update_iter_alpha_in_state!(state::AdaMaxState, method::AdaMax)

    state.iter = state.iter + 1
    state.alpha = method.α(state.iter)
end

function update_state!(d, state::AdaMaxState{T}, method::AdaMax) where {T}
    _update_iter_alpha_in_state!(state, method)
    value_gradient!(d, state.x)
    α, β₁, β₂, ϵ = state.alpha, method.β₁, method.β₂, method.ϵ
    a = 1 - β₁
    m, u = state.m, state.u

    m .= β₁ .* m .+ a .* gradient(d)
    u .= max.(ϵ, max.(β₂ .* u, abs.(gradient(d)))) # I know it's not there in the paper but if m and u start at 0 for some element... NaN occurs next

    @. state.x = state.x - (α / (1 - β₁^state.iter)) * m / u
    # Update current position # x = x + alpha * s
    false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::AdaMax, options, curr_time = time())
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end
