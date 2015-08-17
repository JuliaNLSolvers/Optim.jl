function rosenbrock{T}(x::Vector{T})
    o = one(T)
    c = convert(T,100)
    return (o - x[1])^2 + c * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!{T}(x::Vector{T}, storage::Vector{T})
    o = one(T)
    c = convert(T,100)
    storage[1] = (-2*o) * (o - x[1]) - (4*c) * (x[2] - x[1]^2) * x[1]
    storage[2] = (2*c) * (x[2] - x[1]^2)
end

function rosenbrock_hessian!{T}(x::Vector{T}, storage::Matrix{T})
    o = one(T)
    c = convert(T,100)
    f = 4*c
    storage[1, 1] = (2*o) - f * x[2] + 3 * f * x[1]^2
    storage[1, 2] = -f * x[1]
    storage[2, 1] = -f * x[1]
    storage[2, 2] = 2*c
end

d2 = DifferentiableFunction(rosenbrock,
                            rosenbrock_gradient!)
d3 = TwiceDifferentiableFunction(rosenbrock,
                                 rosenbrock_gradient!,
                                 rosenbrock_hessian!)

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
               :momentum_gradient_descent,
#                :accelerated_gradient_descent,
               :l_bfgs)
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
