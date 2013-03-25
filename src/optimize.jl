# Wrap other functions.
# Automatically wrap inputs and convert vectors appropriately.
function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Vector,
                  method::Symbol,
                  tolerance::Float64,
                  minimize::Bool)
    if method == :nelder_mead
        nelder_mead(f, initial_x)
    elseif method == :simulated_annealing
        simulated_annealing(f, initial_x)
    elseif method == :naive_gradient_descent
        naive_gradient_descent(f, g!, initial_x)
    elseif method == :gradient_descent
        gradient_descent(f, g!, initial_x)
    elseif method == :newton
        newton(f, g!, h!, initial_x)
    elseif method == :bfgs
        bfgs(f, g!, initial_x)
    elseif method == :l_bfgs
        l_bfgs(f, g!, initial_x)
    end
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Vector,
                  method::Symbol)
    if method == :nelder_mead
        nelder_mead(f, initial_x)
    elseif method == :simulated_annealing
        simulated_annealing(f, initial_x)
    elseif method == :naive_gradient_descent
        naive_gradient_descent(f, g!, initial_x)
    elseif method == :gradient_descent
        gradient_descent(f, g!, initial_x)
    elseif method == :newton
        newton(f, g!, h!, initial_x)
    elseif method == :bfgs
        bfgs(f, g!, initial_x)
    elseif method == :l_bfgs
        l_bfgs(f, g!, initial_x)
    else
        error("Unknown method $method")
    end
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Vector)
    newton(f, g!, h!, initial_x)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Vector)
    l_bfgs(d, initial_x)
end

function optimize(f::Function, g!::Function, initial_x::Vector, method::Symbol)
    d = DifferentiableFunction(f, g!)
    if method == :l_bfgs
        l_bfgs(d, initial_x)
    elseif method == :gradient_descent
        gradient_descent(f, g, initial_x)
    elseif method == :naive_gradient_descent
        naive_gradient_descent(d, initial_x)
    else
        error("Unknown method $method")
    end
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Vector)
    l_bfgs(f, g!, initial_x)
end

function optimize(f::Function, initial_x::Vector, method::Symbol)
    if method == :nelder_mead || method == :simulated_annealing
        if method == :nelder_mead
            nelder_mead(f, initial_x)
        else
            simulated_annealing(f, initial_x)
        end
    else
        d = DifferentiableFunction(f)
        if method == :l_bfgs
            l_bfgs(d, initial_x)
        elseif method == :gradient_descent
            gradient_descent(f, g, initial_x)
        elseif method == :naive_gradient_descent
            naive_gradient_descent(d, initial_x)
        else
            error("Unknown method $method")
        end
    end
end

function optimize(f::Function, initial_x::Vector)
    nelder_mead(f, initial_x)
end
