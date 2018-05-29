function backtrack_constrained(ϕ, α::Real, αmax::Real, αImax::Real,
                               Lcoefsα::Tuple{<:Real,<:Real,<:Real}, c1::Real = 0.5,
                               ρ::Real=oftype(α, 0.5),
                               αminfrac::Real = sqrt(eps(one(α)));
                               show_linesearch::Bool=false)
    # TODO: Specify that all elements should be of the same type T <: Real?
    # TODO: What does αI do??
    α, αI = min(α, 0.999*αmax), min(α, 0.999*αImax)
    αmin = αminfrac * α
    L0, L1, L2 = Lcoefsα
    if show_linesearch
        println("L0 = $L0, L1 = $L1, L2 = $L2")
    end
    while α >= αmin
        val = ϕ((α, αI))
        δ = evalgrad(L1, α, αI)
        if show_linesearch
            println("α = $α, αI = $αI, value: ($L0, $val, $(L0+δ))")
        end
        if isfinite(val) && val - (L0 + δ) <= c1*abs(val-L0)
            return α, αI
        end
        α *= ρ
        αI *= ρ
    end
    ϕ((zero(α), zero(αI)))  # to ensure that state gets set appropriately
    return zero(α), zero(αI)
end

function backtrack_constrained_grad(ϕ, α::Real, αmax::Real, Lcoefsα::Tuple{<:Real,<:Real,<:Real},
                                    c1::Real = 0.9, c2::Real = 0.9, ρ::Real=oftype(α, 0.5),
                                    αminfrac::Real = sqrt(eps(one(α))); show_linesearch::Bool=false)
    # TODO: Specify that all elements should be of the same type T <: Real?
    # TODO: Should c1 be 0.9 or 0.5 default?
    # TODO: Should ρ be 0.9 or 0.5 default?
    α = min(α, 0.999*αmax)
    αmin = αminfrac * α
    L0, L1, L2 = Lcoefsα
    if show_linesearch
        println("L0 = $L0, L1 = $L1, L2 = $L2")
    end
    while α >= αmin
        val, slopeα = ϕ(α)
        δval = L1*α
        δslope = L2*α
        if show_linesearch
            println("α = $α, value: ($L0, $val, $(L0+δval)), slope: ($L1, $slopeα, $(L1+δslope))")
        end
        if isfinite(val) && val - (L0 + δval) <= c1*abs(val-L0) &&
            (slopeα < c2*abs(L1) ||
             slopeα - (L1 + δslope) .<= c2*abs.(slopeα-L1))
            return α
        end
        α *= ρ
    end
    ϕ(zero(α))  # to ensure that state gets set appropriately
    return zero(α)
end

# Evaluate for a step parametrized as [α, α, αI, α]
function evalgrad(slopeα, α, αI)
    α*(slopeα[1] + slopeα[2] + slopeα[4]) + αI*slopeα[3]
end

# TODO: Never used anywhere? Intended for a linesearch that depends on ϕ''?
function mulhess(Hα, α, αI)
    αv = [α, α, αI, α]
    Hα*αv
end

# TODO: Never used anywhere? Intended for a linesearch that depends on ϕ''?
function evalhess(Hα, α, αI)
    αv = [α, α, αI, α]
    dot(αv, Hα*αv)
end
