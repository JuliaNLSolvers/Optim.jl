@testset "f increase behaviour" begin
    f(x) = 2*x[1]^2
    g!(G, x) = copy!(G, 4*x[1])


    minimizers = [0.3, -1.5, 0.3, 0.5]
    k = 0
    for allow in [true, false]
        for alpha in [0.1, 1.0]
            k += 1
            method = GradientDescent(linesearch=LineSearches.Static(alpha=alpha))
            opts = Optim.Options(iterations=1,allow_f_increases=allow)
            res = optimize(f, g!, [0.5], method, opts)

            @test minimizers[k] == Optim.minimizer(res)[1]
        end
    end
end
