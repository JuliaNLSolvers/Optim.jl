
function cb(tr::OptimizationTrace)
    @test tr.states[end].iteration % 3 == 0
end

function cb(os::OptimizationState)
    @test os.iteration % 3 == 0
end

for method in (:nelder_mead,
               :simulated_annealing)
    ot_run = false
    function cb(tr::OptimizationTrace)
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(rosenbrock, [0.0,0,.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    function cb(os::OptimizationState)
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(rosenbrock, [0.0,0,.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end

for method in (:bfgs,
               :cg,
               :gradient_descent,
               :momentum_gradient_descent)
    ot_run = false
    function cb(tr::OptimizationTrace)
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(d2, [0.0,0,.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    function cb(os::OptimizationState)
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(d2, [0.0,0,.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end

for method in (:newton,)
    ot_run = false
    function cb(tr::OptimizationTrace)
        @test tr.states[end].iteration % 3 == 0
        ot_run = true
    end
    optimize(d3, [0.0,0.0], method = method, callback = cb, show_every=3, store_trace=true)
    @test ot_run == true

    os_run = false
    function cb(os::OptimizationState)
        @test os.iteration % 3 == 0
        os_run = true
    end
    optimize(d3, [0.0,0.0], method = method, callback = cb, show_every=3)
    @test os_run == true
end
