# Find a stationary solution to the Gross-Pitaevskii equation by
# minimizing the energy of a normalised wave function in a rotating
# frame

@testset "Gross-Pitaveskii vortex" begin
    C = 10
    μ = 10
    Ω = 2*0.575
    h = 0.2
    N = 40

    # Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x
    
    function op(stencil)
        mid = (length(stencil)+1)÷2
        diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
        BandedMatrix(Tuple(diags), (N,N))
    end
    
    ∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    ∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

    y = h/2*(1-N:2:N-1)
    x = y'
    z = Complex.(x,y)
    V = r² = abs2.(z)
    
    init = Complex.(exp.(-r²/2)/√π)
    init = conj.(z).*init./sqrt(1 .+ r²)
    init ./= norm(init)
    
    # The energy that the steady state minimizes
    function E(ψ)
        Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
        sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
    end
    
    # The residual for the GPE, zero in a steady state
    function rdl(ψ)
        Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
        E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
        norm(Lψ-E*ψ)/norm(ψ)
    end
    
    # break out real and imaginary parts, constraint is unit sphere
    function reconstruct(xy)
        n = length(xy) ÷ 2
        ψ = xy[1:n] + 1im*xy[n+1:end]
        m = sqrt(n) |> Int
        ψ = reshape(ψ,m,m)
    end
    
    # norm taken from SOR solution
    cost(xy) = E(20.31*reconstruct(xy))
    
    result = optimize(cost, [real.(init[:]); imag.(init[:])], NelderMead(manifold=Sphere()))
    ψ₂ = 20.31*reconstruct(result.minimizer)
    
    # 3000 steps of successive over-relaxation gets a residual of 1e-3.
    # The default settings in Optim should beat that.
    
    @test_broken rdl(ψ₂) < 1e-3

end