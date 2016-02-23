
for method in (NelderMead(),
               SimulatedAnnealing())
    ot_run = false
    cb = tr -> begin
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(rosenbrock, [0.0,0,.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    cb = os -> begin
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(rosenbrock, [0.0,0,.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end

for method in (BFGS(),
               ConjugateGradient(),
               GradientDescent(),
               MomentumGradientDescent())
    ot_run = false
    cb = tr -> begin
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(d2, [0.0,0,.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    cb = os -> begin
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(d2, [0.0,0,.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end

for method in (Newton(),)
    ot_run = false
    cb = tr -> begin
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(d3, [0.0,0.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    cb = os -> begin
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(d3, [0.0,0.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end
