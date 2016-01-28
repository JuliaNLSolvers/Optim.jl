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
                   nargs...) optimize(d, initial_x, BFGS(linesearch! = linesearch!), OptimizationOptions(nargs...), initial_invH = initial_invH)


@deprecate l_bfgs{T}(d::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                   initial_x::Vector{T};
                   m::Integer = 10,
                   linesearch!::Function = hz_linesearch!,
                   nargs...) optimize(d, initial_x, LBFGS(m = m, linesearch! = linesearch!), OptimizationOptions(nargs...))
