@deprecate NelderMead(a::Real, g::Real, b::Real) NelderMead(initial_simplex = AffineSimplexer(), parameters = FixedParameters(a, g, b, 0.5))
