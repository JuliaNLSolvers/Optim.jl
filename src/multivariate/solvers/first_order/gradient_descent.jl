struct GradientDescent{IL, L, T, Tprep<:Union{Function, Void}} <: FirstOrderSolver
    alphaguess!::IL
    linesearch!::L
    P::T
    precondprep!::Tprep
    manifold::Manifold
end

Base.summary(::GradientDescent) = "Gradient Descent"

"""
# Gradient Descent
## Constructor
```julia
GradientDescent(; alphaguess = LineSearches.InitialHagerZhang(),
linesearch = LineSearches.HagerZhang(),
P = nothing,
precondprep = (P, x) -> nothing)
```
Keywords are used to control choice of line search, and preconditioning.

## Description
The `GradientDescent` method a simple gradient descent algorithm, that is the
search direction is simply the negative gradient at the current iterate, and
then a line search step is used to compute the final step. See Nocedal and
Wright (ch. 2.2, 1999) for an explanation of the approach.

## References
 - Nocedal, J. and Wright, S. J. (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function GradientDescent(; alphaguess = LineSearches.InitialPrevious(), # TODO: Investigate good defaults.
                           linesearch = LineSearches.HagerZhang(),      # TODO: Investigate good defaults
                           P = nothing,
                           precondprep = (P, x) -> nothing,
                           manifold::Manifold=Flat())
    GradientDescent(alphaguess, linesearch, P, precondprep, manifold)
end

mutable struct GradientDescentState{T,N}
    x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state(method::GradientDescent, options, d, initial_x::Array{T}) where T
    initial_x = copy(initial_x)
    retract!(method.manifold, real_to_complex(d,initial_x))

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, real_to_complex(d,gradient(d)), real_to_complex(d,initial_x))

    GradientDescentState(initial_x, # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!(d, state::GradientDescentState{T}, method::GradientDescent) where T
    value_gradient!(d, state.x)
    # Search direction is always the negative preconditioned gradient
    project_tangent!(method.manifold, real_to_complex(d,gradient(d)), real_to_complex(d,state.x))
    method.precondprep!(method.P, real_to_complex(d,state.x))
    A_ldiv_B!(real_to_complex(d,state.s), method.P, real_to_complex(d,gradient(d)))
    scale!(state.s,-1)
    if method.P != nothing
        project_tangent!(method.manifold, real_to_complex(d,state.s), real_to_complex(d,state.x))
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s
    retract!(method.manifold, real_to_complex(d,state.x))
    lssuccess == false # break on linesearch error
end

function assess_convergence(state::GradientDescentState, d, options)
  default_convergence_assessment(state, d, options)
end

function trace!(tr, d, state, iteration, method::GradientDescent, options)
  common_trace!(tr, d, state, iteration, method, options)
end
