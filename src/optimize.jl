function optimize(d::TwiceDifferentiableFunction,
                  initial_x::Vector;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = hz_linesearch!)
    if extended_trace
        store_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :gradient_descent
        gradient_descent(d,
                         initial_x,
                         xtol = xtol,
                         ftol = ftol,
                         grtol = grtol,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace,
                         extended_trace = extended_trace,
                         linesearch! = linesearch!)
    elseif method == :momentum_gradient_descent
        momentum_gradient_descent(d,
                                  initial_x,
                                  xtol = xtol,
                                  ftol = ftol,
                                  grtol = grtol,
                                  iterations = iterations,
                                  store_trace = store_trace,
                                  show_trace = show_trace,
                                  extended_trace = extended_trace,
                                  linesearch! = linesearch!)
    elseif method == :cg
        cg(d,
           initial_x,
           xtol = xtol,
           ftol = ftol,
           grtol = grtol,
           iterations = iterations,
           store_trace = store_trace,
           show_trace = show_trace,
           extended_trace = extended_trace,
           linesearch! = linesearch!)
    elseif method == :bfgs
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!)
    elseif method == :l_bfgs
        l_bfgs(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    elseif method == :newton
        newton(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(d::DifferentiableFunction,
                  initial_x::Vector;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = hz_linesearch!)
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :gradient_descent
        gradient_descent(d,
                         initial_x,
                         xtol = xtol,
                         ftol = ftol,
                         grtol = grtol,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace,
                         extended_trace = extended_trace,
                         linesearch! = linesearch!)
    elseif method == :momentum_gradient_descent
        momentum_gradient_descent(d,
                                  initial_x,
                                  xtol = xtol,
                                  ftol = ftol,
                                  grtol = grtol,
                                  iterations = iterations,
                                  store_trace = store_trace,
                                  show_trace = show_trace,
                                  extended_trace = extended_trace,
                                  linesearch! = linesearch!)
    elseif method == :cg
        cg(d,
           initial_x,
           xtol = xtol,
           ftol = ftol,
           grtol = grtol,
           iterations = iterations,
           store_trace = store_trace,
           show_trace = show_trace,
           extended_trace = extended_trace,
           linesearch! = linesearch!)
    elseif method == :bfgs
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!)
    elseif method == :l_bfgs
        l_bfgs(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Vector;
                  method::Symbol = :newton,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = hz_linesearch!,
                  autodiff::Bool = false)
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :nelder_mead
        nelder_mead(f,
                    initial_x,
                    ftol = ftol,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            extended_trace = extended_trace)
    elseif method == :gradient_descent
        d = DifferentiableFunction(f, g!)
        gradient_descent(d,
                         initial_x,
                         xtol = xtol,
                         ftol = ftol,
                         grtol = grtol,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace,
                         extended_trace = extended_trace,
                         linesearch! = linesearch!)
    elseif method == :momentum_gradient_descent
        d = DifferentiableFunction(f, g!)
        momentum_gradient_descent(d,
                                  initial_x,
                                  xtol = xtol,
                                  ftol = ftol,
                                  grtol = grtol,
                                  iterations = iterations,
                                  store_trace = store_trace,
                                  show_trace = show_trace,
                                  extended_trace = extended_trace,
                                  linesearch! = linesearch!)
    elseif method == :cg
        d = DifferentiableFunction(f, g!)
        cg(d,
           initial_x,
           xtol = xtol,
           ftol = ftol,
           grtol = grtol,
           iterations = iterations,
           store_trace = store_trace,
           show_trace = show_trace,
           extended_trace = extended_trace,
           linesearch! = linesearch!)
    elseif method == :newton
        d = TwiceDifferentiableFunction(f, g!, h!)
        newton(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    elseif method == :bfgs
        d = DifferentiableFunction(f, g!)
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!)
    elseif method == :l_bfgs
        d = DifferentiableFunction(f, g!)
        l_bfgs(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Vector;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = hz_linesearch!,
                  autodiff::Bool = false)
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :nelder_mead
        nelder_mead(f,
                    initial_x,
                    ftol = ftol,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            extended_trace = extended_trace)
    elseif method == :gradient_descent
        d = DifferentiableFunction(f, g!)
        gradient_descent(d,
                         initial_x,
                         xtol = xtol,
                         ftol = ftol,
                         grtol = grtol,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace,
                         extended_trace = extended_trace,
                         linesearch! = linesearch!)
    elseif method == :momentum_gradient_descent
        d = DifferentiableFunction(f, g!)
        momentum_gradient_descent(d,
                                  initial_x,
                                  xtol = xtol,
                                  ftol = ftol,
                                  grtol = grtol,
                                  iterations = iterations,
                                  store_trace = store_trace,
                                  show_trace = show_trace,
                                  extended_trace = extended_trace,
                                  linesearch! = linesearch!)
    elseif method == :cg
        d = DifferentiableFunction(f, g!)
        cg(d,
           initial_x,
           xtol = xtol,
           ftol = ftol,
           grtol = grtol,
           iterations = iterations,
           store_trace = store_trace,
           show_trace = show_trace,
           extended_trace = extended_trace,
           linesearch! = linesearch!)
    elseif method == :bfgs
        d = DifferentiableFunction(f, g!)
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!)
    elseif method == :l_bfgs
        d = DifferentiableFunction(f, g!)
        l_bfgs(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  initial_x::Vector;
                  method::Symbol = :nelder_mead,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = hz_linesearch!,
                  ad::Bool = false)
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :nelder_mead
        return nelder_mead(f,
                    initial_x,
                    ftol = ftol,
                    iterations = iterations,
                    store_trace = store_trace,
                    show_trace = show_trace,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        return simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            extended_trace = extended_trace)
    end
    # otherwise we need a gradient:
    if !ad
        d = DifferentiableFunction(f)
    else
        d = autodiff(f,eltype(initial_x),length(initial_x))
    end
    if method == :gradient_descent
        gradient_descent(d,
                         initial_x,
                         xtol = xtol,
                         ftol = ftol,
                         grtol = grtol,
                         iterations = iterations,
                         store_trace = store_trace,
                         show_trace = show_trace,
                         extended_trace = extended_trace,
                         linesearch! = linesearch!)
    elseif method == :momentum_gradient_descent
        momentum_gradient_descent(d,
                                  initial_x,
                                  xtol = xtol,
                                  ftol = ftol,
                                  grtol = grtol,
                                  iterations = iterations,
                                  store_trace = store_trace,
                                  show_trace = show_trace,
                                  extended_trace = extended_trace,
                                  linesearch! = linesearch!)
    elseif method == :cg
        cg(d,
           initial_x,
           xtol = xtol,
           ftol = ftol,
           grtol = grtol,
           iterations = iterations,
           store_trace = store_trace,
           show_trace = show_trace,
           extended_trace = extended_trace,
           linesearch! = linesearch!)
    elseif method == :bfgs
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!)
    elseif method == :l_bfgs
        l_bfgs(d,
               initial_x,
               xtol = xtol,
               ftol = ftol,
               grtol = grtol,
               iterations = iterations,
               store_trace = store_trace,
               show_trace = show_trace,
               extended_trace = extended_trace,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end
