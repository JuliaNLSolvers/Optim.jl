function backtrack_constrained(ϕ, α, αmax, αImax, Lcoefsα,
                               c1 = 0.5, ρ=oftype(α, 0.5), αminfrac = sqrt(eps(one(α))))
    α, αI = min(α, 0.999*αmax), min(α, 0.999*αImax)
    αmin = αminfrac * α
    L0, L1, L2 = Lcoefsα
    f_calls = 0
    while α >= αmin
        f_calls += 1
        val = ϕ((α, αI))
        δ = evalgrad(L1, α, αI)
        if isfinite(val) && val - (L0 + δ) <= c1*abs(val-L0)
            return α, αI, f_calls, 0
        end
        α *= ρ
        αI *= ρ
    end
    ϕ((zero(α), zero(αI)))  # to ensure that state gets set appropriately
    return zero(α), zero(αI), f_calls, 0
end

function backtrack_constrained_grad(ϕ, α, αmax, Lcoefsα,
                                    c1 = 0.9, c2 = 0.9, ρ=oftype(α, 0.5),
                                    αminfrac = sqrt(eps(one(α))); show_linesearch::Bool=false)
    α = min(α, 0.999*αmax)
    αmin = αminfrac * α
    L0, L1, L2 = Lcoefsα
    if show_linesearch
        println("L0 = $L0, L1 = $L1, L2 = $L2")
    end
    f_calls = 0
    while α >= αmin
        f_calls += 1
        val, slopeα = ϕ(α)
        δval = L1*α
        δslope = L2*α
        if show_linesearch
            println("α = $α, value: ($L0, $val, $(L0+δval)), slope: ($L1, $slopeα, $(L1+δslope))")
        end
        if isfinite(val) && val - (L0 + δval) <= c1*abs(val-L0) &&
            (slopeα < c2*abs(L1) ||
             slopeα - (L1 + δslope) .<= c2*abs.(slopeα-L1))
            return α, f_calls, f_calls
        end
        α *= ρ
    end
    ϕ(zero(α))  # to ensure that state gets set appropriately
    return zero(α), f_calls, f_calls
end

# Evaluate for a step parametrized as [α, α, αI, α]
function evalgrad(slopeα, α, αI)
    α*(slopeα[1] + slopeα[2] + slopeα[4]) + αI*slopeα[3]
end

function mulhess(Hα, α, αI)
    αv = [α, α, αI, α]
    Hα*αv
end
function evalhess(Hα, α, αI)
    αv = [α, α, αI, α]
    dot(αv, Hα*αv)
end
