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
