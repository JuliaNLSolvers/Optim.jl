struct utility_function{T} end

function utility_function{T}(n::Integer) where T
    u=zeros(T,n)
    @. u=max(0,log(n/2+1)-log(1:n))
    su=sum(u)
    @. u=u./su .- 1/n
    u
end

function randchisq(rn::AbstractRNG,::Type{T},ν::Integer) where T
    ν < 0 && throw(ArgumentError("ν (DOF) must be atleast 0."))
    s=zero(T)
    for i in 1:ν
        s+=abs2(randn(rn,T))
    end
    s
end

function randchisq(::Type{T},ν::Integer) where T
    randchisq(Random.default_rng(),T,ν)
end

function geo_mean(x::AbstractVector{T}) where T
    gm=one(T)
    ds=0
    for e in x
        e<=0 && continue
        ds+=1
        gm*=e
    end
    ds==0 && return zero(T)
    gm^(1/ds)
end
