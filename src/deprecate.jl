const method_lookup = Dict{Symbol, Type}(
  :gradient_descent => GradientDescent,
  :momentum_gradient_descent => MomentumGradientDescent,
  :cg => ConjugateGradient,
  :bfgs => BFGS,
  :l_bfgs => LBFGS,
  :newton => Newton,
  :nelder_mead => NelderMead,
  :simulated_annealing => SimulatedAnnealing,
  :brent => Brent,
  :golden_section => GoldenSection,
  :accelerated_gradient_descent => AcceleratedGradientDescent,
  :fminbox => Fminbox)

function get_optimizer(method::Symbol)
    T = method_lookup[method]
    warn("Specifying the method using symbols is deprecated. Use \"method = $(T)()\" instead")
    T()
end

@deprecate bfgs{T}(d::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                   initial_x::Vector{T};
                   initial_invH::Matrix = eye(length(initial_x)),
                   linesearch!::Function = hz_linesearch!,
                   nargs...) optimize(d, initial_x, BFGS(linesearch! = linesearch!), OptimizationOptions(;nargs...), initial_invH = initial_invH)

@deprecate l_bfgs{T}(d::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                   initial_x::Vector{T};
                   m::Integer = 10,
                   linesearch!::Function = hz_linesearch!,
                   nargs...) optimize(d, initial_x, LBFGS(m = m, linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate cg{T}(df::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                initial_x::Array{T};
                linesearch!::Function = hz_linesearch!,
                eta::Real = convert(T,0.4),
                P::Any = nothing,
                precondprep::Function = (P, x) -> nothing,
                nargs...) optimize(df, initial_x, ConjugateGradient(eta = eta, precondprep = precondprep, P = P, linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate gradient_descent{T}(df::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                initial_x::Array{T};
                linesearch!::Function = hz_linesearch!,
                nargs...) optimize(df, initial_x, GradientDescent(linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate momentum_gradient_descent{T}(df::Union{DifferentiableFunction,
                                                  TwiceDifferentiableFunction},
                initial_x::Array{T};
                linesearch!::Function = hz_linesearch!,
                mu::Real = 0.01,
                nargs...) optimize(df, initial_x, MomentumGradientDescent(mu = mu, linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate accelerated_gradient_descent{T}(df::Union{DifferentiableFunction,
                                                     TwiceDifferentiableFunction},
                initial_x::Array{T};
                linesearch!::Function = hz_linesearch!,
                nargs...) optimize(df, initial_x, AcceleratedGradientDescent(linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate newton{T}(df::TwiceDifferentiableFunction,
                initial_x::Array{T};
                linesearch!::Function = hz_linesearch!,
                nargs...) optimize(df, initial_x, Newton(linesearch! = linesearch!), OptimizationOptions(;nargs...))

@deprecate brent{T <: AbstractFloat}(f::Function, x_lower::T, x_upper::T;
                nargs...) optimize(f, x_lower, x_upper, Brent(); nargs...)

@deprecate simulated_annealing{T}(cost::Function,
                                initial_x::Array{T};
                                neighbor!::Function = default_neighbor!,
                                temperature::Function = log_temperature,
                                keep_best::Bool = true,
                                nargs...) optimize(cost, initial_x, SimulatedAnnealing(neighbor! = neighbor!, temperature = temperature, keep_best = keep_best), OptimizationOptions(;nargs...))

@deprecate nelder_mead{T}(f::Function,
                        initial_x::Vector{T};
                        a::Real = 1.0,
                        g::Real = 2.0,
                        b::Real = 0.5,
                        initial_step::Vector{T} = ones(T,length(initial_x)),
                        nargs...) optimize(f, initial_x, NelderMead(a=a, b=b, g=g), OptimizationOptions(nargs...), initial_step = initial_step)

@deprecate golden_section{T <: AbstractFloat}(f::Function, x_lower::T, x_upper::T;
                        nargs...) optimize(f, x_lower, x_upper, GoldenSection(); nargs...)

@deprecate fminbox{T<:AbstractFloat}(df::DifferentiableFunction,
                    initial_x::Array{T}, l::Array{T}, u::Array{T};
                    optimizer = cg, nargs...) optimize(df, initial_x, l, u, Fminbox(); optimizer = get_optimizer(symbol(optimizer)), nargs...)

