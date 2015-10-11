function optimize(d::TwiceDifferentiableFunction,
                  initial_x::Array;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  linesearch!::Function = hz_linesearch!,
                  bfgs_initial_invH = nothing)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
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
                         show_every = show_every,
                         callback = callback,
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
                                  show_every = show_every,
                                  callback = callback,
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
           show_every = show_every,
           callback = callback,
           linesearch! = linesearch!)
    elseif method == :bfgs
        if bfgs_initial_invH == nothing
            bfgs_initial_invH = eye(length(initial_x))
        end
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!,
             show_every = show_every,
             callback = callback,
             initial_invH = bfgs_initial_invH)
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
               show_every = show_every,
               callback = callback,
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  linesearch!::Function = hz_linesearch!,
                  bfgs_initial_invH = nothing)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
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
                         show_every = show_every,
                         callback = callback,
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
                                  show_every = show_every,
                                  callback = callback,
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
           show_every = show_every,
           callback = callback,
           linesearch! = linesearch!)
    elseif method == :bfgs
        if bfgs_initial_invH == nothing
            bfgs_initial_invH = eye(length(initial_x))
        end
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             linesearch! = linesearch!,
             show_every = show_every,
             callback = callback,
             initial_invH = bfgs_initial_invH)
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Array;
                  method::Symbol = :newton,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  linesearch!::Function = hz_linesearch!,
                  bfgs_initial_invH = nothing)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
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
                    show_every = show_every,
                    callback = callback,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            show_every = show_every,
                            callback = callback,
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
                         show_every = show_every,
                         callback = callback,
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
                                  show_every = show_every,
                                  callback = callback,
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
           show_every = show_every,
           callback = callback,
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    elseif method == :bfgs
        if bfgs_initial_invH == nothing
            bfgs_initial_invH = eye(length(initial_x))
        end
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
             show_every = show_every,
             callback = callback,
             linesearch! = linesearch!,
             initial_invH = bfgs_initial_invH)
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Array;
                  method::Symbol = :l_bfgs,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  linesearch!::Function = hz_linesearch!,
                  bfgs_initial_invH = nothing)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
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
                    show_every = show_every,
                    callback = callback,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            show_every = show_every,
                            callback = callback,
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
                         show_every = show_every,
                         callback = callback,
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
                                  show_every = show_every,
                                  callback = callback,
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
           show_every = show_every,
           callback = callback,
           linesearch! = linesearch!)
    elseif method == :bfgs
        if bfgs_initial_invH == nothing
            bfgs_initial_invH = eye(length(initial_x))
        end
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
             show_every = show_every,
             callback = callback,
             linesearch! = linesearch!,
             initial_invH = bfgs_initial_invH)
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  initial_x::Array;
                  method::Symbol = :nelder_mead,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  linesearch!::Function = hz_linesearch!,
                  autodiff::Bool = false,
                  bfgs_initial_invH = nothing)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
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
                    show_every = show_every,
                    callback = callback,
                    extended_trace = extended_trace)
    elseif method == :simulated_annealing
        return simulated_annealing(f,
                            initial_x,
                            iterations = iterations,
                            store_trace = store_trace,
                            show_trace = show_trace,
                            show_every = show_every,
                            callback = callback,
                            extended_trace = extended_trace)
    end
    # otherwise we need a gradient:
    if !autodiff
        d = DifferentiableFunction(f)
    else
        d = Optim.autodiff(f, eltype(initial_x), length(initial_x))
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
                         show_every = show_every,
                         callback = callback,
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
                                  show_every = show_every,
                                  callback = callback,
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
           show_every = show_every,
           callback = callback,
           linesearch! = linesearch!)
    elseif method == :bfgs
        if bfgs_initial_invH == nothing
            bfgs_initial_invH = eye(length(initial_x))
        end
        bfgs(d,
             initial_x,
             xtol = xtol,
             ftol = ftol,
             grtol = grtol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             extended_trace = extended_trace,
             show_every = show_every,
             callback = callback,
             linesearch! = linesearch!,
             initial_invH = bfgs_initial_invH)
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
               show_every = show_every,
               callback = callback,
               linesearch! = linesearch!)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize{T <: AbstractFloat}(f::Function,
                                      lower::T,
                                      upper::T;
                                      method::Symbol = :brent,
                                      rel_tol::Real = sqrt(eps(T)),
                                      abs_tol::Real = eps(T),
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      callback = nothing,
                                      show_every = 1,
                                      extended_trace::Bool = false)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    if method == :brent
        brent(f, Float64(lower), Float64(upper);
              rel_tol = rel_tol,
              abs_tol = abs_tol,
              iterations = iterations,
              store_trace = store_trace,
              show_trace = show_trace,
              show_every = show_every,
              callback = callback,
              extended_trace = extended_trace)
    elseif method == :golden_section
        golden_section(f, Float64(lower), Float64(upper);
                       rel_tol = rel_tol,
                       abs_tol = abs_tol,
                       iterations = iterations,
                       store_trace = store_trace,
                       show_trace = show_trace,
                       show_every = show_every,
                       callback = callback,
                       extended_trace = extended_trace)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function optimize(f::Function,
                  lower::Real,
                  upper::Real;
                  kwargs...)
    optimize(f,
             Float64(lower),
             Float64(upper);
             kwargs...)
end
