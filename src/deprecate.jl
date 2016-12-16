@deprecate NelderMead(a::Real, g::Real, b::Real) NelderMead(initial_simplex = AffineSimplexer(), parameters = FixedParameters(a, g, b, 0.5))

@deprecate MultivariateOptimizationResults(method,
    initial_x, minimizer, minimum, iterations,
    iteration_converged, x_converged, x_tol,
    f_converged, f_tol, g_converged, g_tol,
    trace, f_calls, g_calls)  MultivariateOptimizationResults(method,
        initial_x, minimizer, minimum, iterations,
        iteration_converged, x_converged, x_tol,
        f_converged, f_tol, g_converged, g_tol,
        trace, f_calls, g_calls, 0)

# LineSearches deprecation
for name in names(LineSearches)
    if name in (:LineSearches, :hz_linesearch!, :mt_linesearch!,
                :interpolating_linesearch!, :backtracking_linesearch!,
                :interpbacktracking_linesearch!)
        continue
    end
    eval(:(@deprecate $name LineSearches.$name))
end

@deprecate hz_linesearch! LineSearches.hagerzhang!
@deprecate mt_linesearch! LineSearches.morethuente!
@deprecate interpolating_linesearch! LineSearches.strongwolfe!
@deprecate backtracking_linesearch! LineSearches.backtracking!
@deprecate interpbacktracking_linesearch! LineSearches.interpbacktracking!


# Multivariate optimization
@deprecate optimize{F<:Function}(f::F,
                  initial_x::Array;
                  method = NelderMead(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  autodiff::Bool = false,
                  callback = nothing) optimize(f, initial_x, method, Options(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every,
        autodiff = autodiff))

@deprecate optimize{F<:Function, G<:Function}(f::F,
                  g!::G,
                  initial_x::Array;
                  method = LBFGS(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing) optimize(f, g!, initial_x, method, Options(;
                      x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
                      iterations = iterations, store_trace = store_trace,
                      show_trace = show_trace, extended_trace = extended_trace,
                      callback = callback, show_every = show_every))

@deprecate optimize{F<:Function, G<:Function, H<:Function}(f::F,
                  g!::G,
                  h!::H,
                  initial_x::Array;
                  method = Newton(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing) optimize(f, g!, h!, initial_x, method, Options(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every))


@deprecate optimize(d::DifferentiableFunction,
                          initial_x::Array;
                          method = LBFGS(),
                          x_tol::Real = 1e-32,
                          f_tol::Real = 1e-32,
                          g_tol::Real = 1e-8,
                          iterations::Integer = 1_000,
                          store_trace::Bool = false,
                          show_trace::Bool = false,
                          extended_trace::Bool = false,
                          show_every::Integer = 1,
                          callback = nothing) optimize(d, initial_x, method, Options(;
                              x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
                              iterations = iterations, store_trace = store_trace,
                              show_trace = show_trace, extended_trace = extended_trace,
                              callback = callback, show_every = show_every))

@deprecate optimize(d::TwiceDifferentiableFunction,
                          initial_x::Array;
                          method = Newton(),
                          x_tol::Real = 1e-32,
                          f_tol::Real = 1e-32,
                          g_tol::Real = 1e-8,
                          iterations::Integer = 1_000,
                          store_trace::Bool = false,
                          show_trace::Bool = false,
                          extended_trace::Bool = false,
                          show_every::Integer = 1,
                          callback = nothing) optimize(d, initial_x, method, Options(;
                              x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
                              iterations = iterations, store_trace = store_trace,
                              show_trace = show_trace, extended_trace = extended_trace,
                              callback = callback, show_every = show_every))

@deprecate OptimizationOptions(args...; kwargs...) Optim.Options(args...; kwargs...)
