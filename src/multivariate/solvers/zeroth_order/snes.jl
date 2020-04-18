struct sNES <: ZerothOrderOptimizer
end

mutable struct sNESstate{T,Tx,Tfs} <: ZerothOrderState
    d::Int
    ημ::T
    ησ::T
    samples::Int
    x::Tx
    σ::Tx
    tmp_x::Tx
    F::Tfs
    ∇f::Tx
    ∇fσ::Tx
    ϵ::Vector{Tx}
    u::Tx
    idx::Vector{Int}
    iteration::Int
end

function initial_state(method::sNES, options, f, x0::AbstractVector{T}) where T
    d=length(x0)
    ημ=T(1)
    ησ=T( (3+log(d))/(5*√d) )
    samples=4 + ceil(Int, log(3*d))
    N=length(x0)
    x = copy(x0)
    tmp_x = copy(x0)
    ∇f = copy(x0)
    ∇fσ = copy(x0)
    ϵ = map(i->copy(x0), 1:samples)
    F = fill(value(f,x0),samples)
    u = utility_function{T}(samples)
    idx = collect(1:samples)
    σ=fill!(copy(x0),1)
    for i in 1:samples
        randn!(ϵ[i])
        tmp_x .= x .+ σ .* ϵ[i]
        F[i] = value(f,tmp_x)
    end
    sortperm!(idx,F)
    sNESstate{T,typeof(x),typeof(F)}(d,ημ,ησ,samples,x,σ,tmp_x,F,∇f,∇fσ,ϵ,u,idx,1)
end

function  update_state!(f, state::sNESstate{T}, method::sNES) where T
    Z = zero(T)
    O = one(T)

    for i in 1:state.samples
        j = state.idx[i]
        if i >= (state.samples ÷ 2)
            randn!(state.ϵ[j])
        else
            rescaling = sqrt(randchisq(T, state.d)) / norm(state.ϵ[j])
            state.ϵ[j] .*= rescaling
        end
        state.tmp_x .= state.x .+ state.σ .* state.ϵ[j]
        state.F[j] = value(f,state.tmp_x)
    end
    sortperm!(state.idx,state.F)
    state.∇f .= Z
    state.∇fσ .= Z
    for i in 1:state.samples
        j = state.idx[i]
        state.∇f .+= state.u[i] .* state.ϵ[j]
        state.∇fσ .+= state.u[i] .* (state.ϵ[j] .* state.ϵ[j] .- O)
    end
    state.x .+= state.ημ .* state.σ .* state.∇f
    state.σ .*= exp.(state.ησ ./ 2 .* state.∇fσ)
    state.iteration+=1
    false
end

pick_best_x(f_increased, state::sNESstate) = state.x
pick_best_f(f_increased, state::sNESstate, d) = value(d,state.x)

function assess_convergence(state::sNESstate, d, options::Options)
    g_converged = geo_mean(state.σ) <= options.g_abstol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, false
end
function default_options(method::sNES)
    Dict(:allow_f_increases => true, :g_abstol=>1e-8)
end
