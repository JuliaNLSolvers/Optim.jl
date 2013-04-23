function optimize(d::DifferentiableFunction,
                  initial_x::Vector;
                  method::Symbol = :l_bfgs,
                  tolerance::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false)
    if method == :gradient_descent
        gradient_descent(d,
                         initial_x,
                         tolerance = tolerance,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace)
    elseif method == :bfgs
        bfgs(d,
             initial_x,
             tolerance = tolerance,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace)
    elseif method == :l_bfgs
        l_bfgs(d,
               initial_x,
               tolerance = tolerance,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Vector;
                  method::Symbol = :newton,
                  tolerance::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false)
    if method == :nelder_mead
        nelder_mead(f,
                    initial_x,
                    tolerance = tolerance,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            tolerance = tolerance,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace)
    elseif method == :gradient_descent
        d = DifferentiableFunction(f, g!)
        gradient_descent(d,
                         initial_x,
                         tolerance = tolerance,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace)
    elseif method == :newton
        d = TwiceDifferentiableFunction(f, g!, h!)
        newton(d,
               initial_x,
               tolerance = tolerance,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace)
    elseif method == :bfgs
        d = DifferentiableFunction(f, g!)
        bfgs(d,
             initial_x,
             tolerance = tolerance,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace)
    elseif method == :l_bfgs
        d = DifferentiableFunction(f, g!)
        l_bfgs(d,
               initial_x,
               tolerance = tolerance,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Vector;
                  method::Symbol = :l_bfgs,
                  tolerance::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false)
    if method == :nelder_mead
        nelder_mead(f,
                    initial_x,
                    tolerance = tolerance,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            tolerance = tolerance,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace)
    elseif method == :gradient_descent
        d = DifferentiableFunction(f, g!)
        gradient_descent(d,
                         initial_x,
                         tolerance = tolerance,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace)
    elseif method == :bfgs
        d = DifferentiableFunction(f, g!)
        bfgs(d,
             initial_x,
             tolerance = tolerance,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace)
    elseif method == :l_bfgs
        d = DifferentiableFunction(f, g!)
        l_bfgs(d,
               initial_x,
               tolerance = tolerance,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  initial_x::Vector;
                  method::Symbol = :nelder_mead,
                  tolerance::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false)
    if method == :nelder_mead
        nelder_mead(f,
                    initial_x,
                    tolerance = tolerance,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            tolerance = tolerance,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace)
    elseif method == :gradient_descent
        d = DifferentiableFunction(f)
        gradient_descent(d,
                         initial_x,
                         tolerance = tolerance,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace)
    elseif method == :bfgs
        d = DifferentiableFunction(f)
        bfgs(d,
             initial_x,
             tolerance = tolerance,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace)
    elseif method == :l_bfgs
        d = DifferentiableFunction(f)
        l_bfgs(d,
               initial_x,
               tolerance = tolerance,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end
