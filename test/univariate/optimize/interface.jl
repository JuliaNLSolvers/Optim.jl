@testset "#853" begin

    "Model parameter"
    struct yObj
        p::Any
    end

    "Univariable Functor"
    function (s::yObj)(x)
        return x * s.p
    end

    "Multivariable Functor with array input"
    function (s::yObj)(x_v)
        return x_v[1] * s.p
    end


    model = yObj(1.0)

    Optim.optimize(model, -1, 2, [1.0]) # already worked
    Optim.optimize(model, -1, 2) # didn't work
    Optim.optimize(model, -1.0, 2.0) # didn't work
end
