struct xNES <: ZerothOrderOptimizer
end

mutable struct xNESstate{T,Tx,Tm,Tfs} <: ZerothOrderState
    d::Int
    ημ::T
    ησ::T
    ηB::T
    σ::T
    samples::Int
    x::Tx
    tmp_x::Tx
    u::Tx
    Gδ::Tx
    F::Tfs
    Z::Tm
    B::Tm
    GM::Tm
    idx::Vector{Int}
    iteration::Int
end

function initial_state(method::xNES, options, f, x0::AbstractVector{T}) where T
    d=length(x0)
    ημ=T(1)
    ηB=ησ=T( (9+3*log(d))/(5*d*√d) )
    samples=4 + ceil(Int, log(3*d))
    x=copy(x0)
    d=length(x)
    Z=Array{T}(undef,d,samples)
    A=zeros(T,d,d)
    for i in 1:d
        A[i,i]=1
    end
    σ=abs(det(A))^(1/d)
    F=fill(value(f,x),samples)
    B=A./σ
    idx=collect(1:samples)
    u=utility_function{T}(samples)
    Gδ=zeros(T,d)
    GM=zeros(T,d,d)
    tmp_x=copy(x)
    iteration=0
    xNESstate{T,typeof(x0),typeof(A),typeof(F)}(d,ημ,ησ,ηB,σ,samples,x,tmp_x,u,Gδ,F,Z,B,GM,idx,iteration)
end

function δ(T,i,j)
    ifelse(i==j,one(T),zero(T))
end

function update_state!(f, state::xNESstate{T}, method::xNES) where T
    randn!(state.Z)
    for i in 1:state.samples
        #F[i] = f(x .+ σ .* B * Z[:,i])
        mul!(state.tmp_x,state.B,view(state.Z,:,i))
        state.tmp_x .= state.x .+ state.tmp_x .* state.σ
        state.F[i] = value(f,state.tmp_x)
    end
    sortperm!(state.idx,state.F)
    #Gδ=sum(u[i] .* Z[:,idx[i]] for i in 1:n)
    #GM=sum(u[i] .* (Z[:,idx[i]] * (Z[:,idx[i]]') - I) for i in 1:n)
    let i=1
        j = state.idx[i]
        for k2 in 1:state.d
            state.Gδ[k2] = state.u[i] * state.Z[k2,j]
            for k1 in 1:state.d
                state.GM[k1,k2] = state.u[i] * (state.Z[k1,j] * state.Z[k2,j] - δ(T,k1,k2))
            end
        end
    end
    for i in 2:state.samples
        j = state.idx[i]
        for k2 in 1:state.d
            state.Gδ[k2] += state.u[i] * state.Z[k2,j]
            for k1 in 1:state.d
                state.GM[k1,k2] += state.u[i] * (state.Z[k1,j] * state.Z[k2,j] - δ(T,k1,k2))
            end
        end
    end
    #=
    Gσ = tr(GM) / d
    GB=GM - Gσ * I
    x=x .+ ημ * σ *B *Gδ
    σ=σ * exp(ησ/2 * Gσ)
    B=B * exp(ηB/2 .* GB)
    =#
    Gσ = tr(state.GM) / state.d
    for i in 1:state.d
        state.GM[i,i]-=Gσ
    end
    state.GM .*= state.ηB/2
    Gσ *= state.ησ/2
    mul!(state.tmp_x,state.B,state.Gδ)
    state.x .+= state.ημ .* state.σ .* state.tmp_x
    state.σ *= exp(Gσ)
    mul!(state.GM,state.B,exp(state.GM))
    state.B.=state.GM
    state.iteration+=1
    false
end

pick_best_x(f_increased, state::xNESstate) = state.x
pick_best_f(f_increased, state::xNESstate, d) = value(d,state.x)

function assess_convergence(state::xNESstate, d, options::Options)
    g_converged = state.σ <= options.g_abstol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, false
end

function default_options(method::xNES)
    Dict(:allow_f_increases => true, :g_abstol=>1e-8)
end
