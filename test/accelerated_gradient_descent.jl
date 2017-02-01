# TODO expand tests here
@testset "Accelerated Gradient Descent" begin 
    f(x) = x[1]^4
    function g!(x, storage)
        storage[1] = 4 * x[1]^3
        return
    end

    d = OnceDifferentiable(f, g!)

    initial_x = [1.0]
    options = Optim.Options(show_trace = true, iterations = 10)
    Optim.optimize(d, initial_x, AcceleratedGradientDescent(), options)
end
