@deprecate NelderMead(a::Real, g::Real, b::Real) NelderMead(parameters = FixedParameters(a, g, b, 0.5))
