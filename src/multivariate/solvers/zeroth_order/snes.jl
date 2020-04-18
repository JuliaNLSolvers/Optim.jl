struct sNES{T}
    ημ::T
    ησ::T
    σtol::T
    samples::Int
    function sNES{T}(d::Integer;P...) where T
        ημ=T(1)
        ησ=T( (3+log(d))/(5*√d) )
        samples=4 + ceil(Int, log(3*d))
        σtol=T(1e-8)
        haskey(P,:ημ) && (ημ=P[:ημ])
        haskey(P,:ησ) && (ησ=P[:ησ])
        haskey(P,:samples) && (samples=P[:samples])
        haskey(P,:σtol) && (σtol=P[:σtol])
        new{T}(ημ,ησ,σtol,samples)
    end
end

function separable_nes(f,x0::AbstractVector{T},σ::AbstractVector{T},params::sNES{T}) where T
    samples=params.samples
    samples>3 || error("atleast 3 samples are required for sNES to work.")
    ημ = params.ημ
    ησ = params.ησ
    σtol = params.σtol
    N=length(x0)
    length(σ) == N || error("the length of 'σ' must be equal to the length of 'x'.")
    x = copy(x0)
    Z = zero(T)
    O = one(T)
    F = fill(f(x0),samples)
    ∇f = fill(Z,N)
    ∇fσ = fill(Z,N)
    ϵ = map(i->zeros(T,N), 1:samples)
    u = utility_function{T}(samples)
    idx = collect(1:samples)
    tmp_x = copy(x0)
    for i in 1:samples
        randn!(ϵ[i])
        tmp_x .= x .+ σ .* ϵ[i]
        F[i] = f(tmp_x)
    end
    sortperm!(idx,F)
    while geo_mean(σ) > σtol
        for i in 1:samples
            j = idx[i]
            if i >= (samples ÷ 2)
                randn!(ϵ[j])
            else
                rescaling = sqrt(randchisq(T, N)) / norm(ϵ[j])
                ϵ[j] .*= rescaling
            end
            tmp_x .= x .+ σ .* ϵ[j]
            F[j] = f(tmp_x)
        end
        sortperm!(idx,F)
        ∇f .= Z
        ∇fσ .= Z
        for i in 1:samples
            j = idx[i]
            ∇f .+= u[i] .* ϵ[j]
            ∇fσ .+= u[i] .* (ϵ[j] .* ϵ[j] .- O)
        end
        x .+= ημ .* σ .* ∇f
        σ .*= exp.(ησ ./ 2 .* ∇fσ)
    end
    (sol = x, cost = f(x))
end

function separable_nes(f,μ::AbstractVector{T},σ::T,params::sNES{T}) where T
    separable_nes(f,μ,fill(σ,length(μ)),params)
end

function separable_nes(f,μ::AbstractVector{T},σ; P...) where T
    separable_nes(f,μ,σ,sNES{T}(length(μ);P...))
end
