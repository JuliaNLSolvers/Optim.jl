function backtrack_constrained(ϕ, α, αmax, αImax, Lcoefsα,
                               c1 = 0.5, ρ=oftype(α, 0.5), αmin = sqrt(eps(one(α))))
    α, αI = min(α, 0.999*αmax), min(α, 0.999*αImax)
    L0, L1, L2 = Lcoefsα
    f_calls = 0
    while α >= αmin
        f_calls += 1
        val = ϕ(α, αI)
        if isfinite(val) && abs(val - (L0 + L1*α + L2*α^2/2)) <= c1*abs(val-L0)
            return α, αI, f_calls, 0
        end
        α *= ρ
        αI *= ρ
    end
    return zero(α), zero(αI), f_calls, 0
end
