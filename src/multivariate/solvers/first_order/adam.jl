"""
# Adam
## Constructor
```julia
    Adam(; alpha=0.0001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8)
```
## Description
Adam is a gradient based optimizer that choses its search direction by building up estimates of the first two moments of the gradient vector. This makes it suitable for problems with a stochastic objective and thus gradient. The method is introduced in [1] where the related AdaMax method is also introduced, see `?AdaMax` for more information on that method.

## References
[1] https://arxiv.org/abs/1412.6980
"""
struct Adam{T, Tm} <: FirstOrderOptimizer
    α::T
    β₁::T
    β₂::T
    ϵ::T
    manifold::Tm
end
Adam(; alpha = 0.0001, beta_mean = 0.9, beta_var = 0.999, epsilon = 1e-8) =
    Adam(alpha, beta_mean, beta_var, epsilon, Flat())
Base.summary(::Adam) = "Adam"
function default_options(method::Adam)
    (; allow_f_increases = true, iterations=10_000)
end

mutable struct AdamState{Tx, T, Tz, Tm, Tu, Ti} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    f_x_previous::T
    s::Tx
    z::Tz
    m::Tm
    u::Tu
    iter::Ti
end
function reset!(method, state::AdamState, obj, x)
    value_gradient!!(obj, x)
end
function initial_state(method::Adam, options, d, initial_x::AbstractArray{T}) where T
    initial_x = copy(initial_x)

    value_gradient!!(d, initial_x)
    α, β₁, β₂ = method.α, method.β₁, method.β₂
   
    z = copy(initial_x)
    m = copy(gradient(d))
    u = fill(zero(m[1]^2), length(m))
    a = 1 - β₁
    iter = 0

    AdamState(initial_x, # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         real(T(NaN)), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         z,
                         m,
                         u,
                         iter)
end

function update_state!(d, state::AdamState{T}, method::Adam) where T
    state.iter = state.iter+1
    value_gradient!(d, state.x)
    α, β₁, β₂, ϵ = method.α, method.β₁, method.β₂, method.ϵ
    a = 1 - β₁
    b = 1 - β₂

    m, u, z = state.m, state.u, state.z
    v = u
    m .= β₁ .* m .+ a .* gradient(d)
    v .= β₂ .* v .+ b .* gradient(d) .^ 2
    #  m̂ = m./(1-β₁^state.iter)
    # v̂ = v./(1-β₂^state.iter)
    #@. z = z - α*m̂/(sqrt(v̂+ϵ))
    @. z = z - α*m/(1-β₁^state.iter)/(sqrt(v./(1-β₂^state.iter)+ϵ))

    # not quite the same because epsilon is in the sqrt
    # not sure where I got this from
    #    αₜ = α * sqrt(1 - β₂^state.iter) / (1 - β₁^state.iter)
    #    z .= z .- αₜ .* m ./ (sqrt.(v .+ ϵ) )

    for _i in eachindex(z)
        # since m and u start at 0, this can happen if the initial gradient is exactly 0
        # rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        # optimize(rosenbrock, zeros(2), Adam(), Optim.Options(iterations=10000))
        if isnan(z[_i])
            z[_i] = state.x[_i]
        end
    end
    state.x .= z
    # Update current position # x = x + alpha * s
    false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::Adam, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end
