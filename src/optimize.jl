function getOptimizer(method::Optimizer)
    method
end

function getOptimizer(method::Symbol)
    T = if method == :gradient_descent
      GradientDescent
    elseif method == :momentum_gradient_descent
      MomentumGradientDescent
    elseif method == :cg
      ConjugateGradient
    elseif method == :bfgs
      BFGS
    elseif method == :l_bfgs
      LBFGS
    elseif method == :newton
      Newton
    elseif method == :nelder_mead
      NelderMead
    elseif method == :simulated_annealing
      SimulatedAnnealing
    elseif method == :brent
      Brent
    elseif method == :golden_section
      GoldenSection
    elseif method == :accelerated_gradient_descent
      AcceleratedGradientDescent
    elseif method == :fminbox
      Fminbox
    else
      throw(ArgumentError("Unknown method $method"))
    end
    warn("Specifying the method using symbols is deprecated. Use \"method = $(T)()\" instead")
    T()
end

function optimize(f::Function,
                  initial_x::Array,
                  method::Optimizer;
                  autodiff::Bool = false,
                  nargs...)
    if !autodiff
        d = DifferentiableFunction(f)
    else
        d = Optim.autodiff(f, eltype(initial_x), length(initial_x))
    end
    optimize(d, initial_x, method;
             nargs...)
end

function optimize(d::TwiceDifferentiableFunction,
                  initial_x::Array,
                  method::Optimizer;
                  nargs...)
    dn = DifferentiableFunction(d.f, d.g!, d.fg!)
    optimize(dn, initial_x, method;
             nargs...)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array,
                  method::Optimizer;
                  nargs...)
    optimize(d.f, initial_x, method;
             nargs...)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array;
                  method = LBFGS(),
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  nargs...)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    method = getOptimizer(method)::Optimizer
    optimize(d, initial_x, method;
             show_every = show_every,
             show_trace = show_trace,
             extended_trace = extended_trace,
             callback = callback,
             nargs...)
end

function optimize(d::TwiceDifferentiableFunction,
                  initial_x::Array;
                  method = LBFGS(),
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  nargs...)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    method = getOptimizer(method)::Optimizer
    optimize(d, initial_x, method;
             show_every = show_every,
             show_trace = show_trace,
             extended_trace = extended_trace,
             callback = callback,
             nargs...)
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Array;
                  method = Newton(),
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  nargs...)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    method = getOptimizer(method)::Optimizer
    d = TwiceDifferentiableFunction(f, g!, h!)
    optimize(d, initial_x, method;
             show_every = show_every,
             show_trace = show_trace,
             extended_trace = extended_trace,
             callback = callback,
             nargs...)
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Array;
                  method = LBFGS(),
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  nargs...)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    method = getOptimizer(method)::Optimizer
    d = DifferentiableFunction(f, g!)
    optimize(d, initial_x, method;
             show_every = show_every,
             show_trace = show_trace,
             extended_trace = extended_trace,
             callback = callback,
             nargs...)
end

function optimize(f::Function,
                  initial_x::Array;
                  method = NelderMead(),
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  callback = nothing,
                  show_every = 1,
                  nargs...)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    method = getOptimizer(method)::Optimizer
    optimize(f, initial_x, method;
             show_every = show_every,
             show_trace = show_trace,
             extended_trace = extended_trace,
             callback = callback,
             nargs...)
end

function optimize{T <: AbstractFloat}(f::Function,
                                      lower::T,
                                      upper::T;
                                      method = Brent(),
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
    method = getOptimizer(method)::Optimizer
    optimize(f, Float64(lower), Float64(upper), method;
             rel_tol = rel_tol,
             abs_tol = abs_tol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             show_every = show_every,
             callback = callback,
             extended_trace = extended_trace)
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
