"""
    AdaMax(; alpha=0.002, beta_mean=0.9, beta_var=0.999)
# Adam
## Constructor
```julia
    AdaMax(; alpha=0.002, beta_mean=0.9, beta_var=0.999)
```
## Description
AdaMax is a gradient based optimizer that choses its search direction by building up estimates of the first two moments of the gradient vector. This makes it suitable for problems with a stochastic objective and thus gradient. The method is introduced in [1] where the related Adam method is also introduced, see `?Adam` for more information on that method.


[1] https://arxiv.org/abs/1412.6980
"""

struct AdaMax{T,Tm} <: FirstOrderOptimizer
    α::T
    β₁::T
    β₂::T
    manifold::Tm
end
AdaMax(; alpha = 0.002, beta_mean = 0.9, beta_var = 0.999) =
    AdaMax(alpha, beta_mean, beta_var, Flat())
Base.summary(::AdaMax) = "AdaMax"
function default_options(method::AdaMax)
    (; allow_f_increases = true, iterations=10_000)
end


mutable struct AdaMaxState{Tx, T, Tz, Tm, Tu, Ti} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    f_x_previous::T
    s::Tx
    z::Tz
    m::Tm
    u::Tu
    iter::Ti
end
function reset!(method, state::AdaMaxState, obj, x)
    value_gradient!!(obj, x)
end
function initial_state(method::AdaMax, options, d, initial_x::AbstractArray{T}) where T
    initial_x = copy(initial_x)

    value_gradient!!(d, initial_x)
    α, β₁, β₂ = method.α, method.β₁, method.β₂
   
    z = copy(initial_x)
    m = copy(gradient(d))
    u = zero(m)
    a = 1 - β₁
    iter = 0

    AdaMaxState(initial_x, # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         real(T(NaN)), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         z,
                         m,
                         u,
                         iter)
end

function update_state!(d, state::AdaMaxState{T}, method::AdaMax) where T
    state.iter = state.iter+1
    value_gradient!(d, state.x)
    α, β₁, β₂ = method.α, method.β₁, method.β₂
    a = 1 - β₁
    m, u, z = state.m, state.u, state.z

    m .= β₁ .* m .+ a .* gradient(d)
    u .= max.(β₂ .* u, abs.(gradient(d)))
    z .= z .- (α ./ (1 - β₁^state.iter)) .* m ./ u
    for _i in eachindex(z)
        # since m and u start at 0, this can happen if the initial gradient is exactly 0
        # rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        # optimize(rosenbrock, zeros(2), AdaMax(), Optim.Options(iterations=10000))
        if isnan(z[_i])
            z[_i] = state.x[_i]
        end
    end
    state.x .= z
    # Update current position # x = x + alpha * s
    false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::AdaMax, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end
