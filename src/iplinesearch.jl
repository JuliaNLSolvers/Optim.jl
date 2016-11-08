function backtrack_constrained(ϕ, α, αmax, Lcoefsα,
                               c1 = 0.5, ρ=oftype(α, 0.5), itermax = 100)
    α = min(α, 0.999*αmax)
    L0, L1, L2 = Lcoefsα
    f_calls = 0
    while f_calls < itermax
        f_calls += 1
        val = ϕ(α)
        if abs(val - (L0 + L1*α + L2*α^2/2)) <= c1*abs(val-L0) + 100*eps(abs(val)+abs(L0))
            return α, f_calls, 0
        end
        α *= ρ
    end
    error("failed to satisfy criterion after $f_calls iterations")
end
