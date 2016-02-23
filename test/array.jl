module TestArray
    using Optim
    using Base.Test

    f(X) = (10 - X[1, 1])^2 + (0 - X[1, 2])^2 + (0 - X[2, 1])^2 + (5 - X[2, 2])^2

    function g!(X, S)
        S[1, 1] = -20 + 2 * X[1, 1]
        S[1, 2] = 2 * X[1, 2]
        S[2, 1] = 2 * X[2, 1]
        S[2, 2] = -10 + 2 * X[2, 2]
        return
    end

    res = optimize(f, g!, eye(2), method = GradientDescent())

    @test norm(vec(res.minimum - [10.0 0.0; 0.0 5.0])) < 10e-8

    # TODO: Get finite differencing to work for generic arrays as well
    # optimize(f, eye(2), method = :gradient_descent)
end
