#Combined Gradient Descent with RMSprop element
#http://www.cs.toronto.edu/~tijmen/csc321/slides/lectureslideslec6.pdf

struct RMSprop{Tf,Tg,Th,Ti, IL,L} <: FirstOrderOptimizer
    mu::Tf
    alpha::Tg
	beta::Th
	epsilon::Ti
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(::RMSprop) = "Root Mean Square propagation"

"""
# Root Mean Square propagation (RMSprop)
## Constructor
```julia
RMSprop(; mu::Real = 0.9,
 alpha::Real = 0.00000001, 
 beta::Real = 0.99, 
 epsilon::Real = 0.000001, 
 alphaguess = LineSearches.InitialPrevious(), 
 linesearch = LineSearches.HagerZhang(),       
```

## Description
RMSProp (for Root Mean Square Propagation) is a method in which the learning rate(here alpha) is adapted for 
each of the parameters. The idea is to divide the learning rate for a weight by a running average of the 
magnitudes of recent gradients for that function.

In this case, RMSprop is combined with simple gradient descent algorithm, which uses a line search step 
to compute the next step.

## Formula
y = y_previous * beta + ( 1 - beta ) * g ^ 2

x = x_previous + mu * a * g + alpha * g / (sqrt(y) + epsilon)

parameters:

a= alpha calculated of line search

g= gradient of recent iteration

## References
 - RMSprop is an unpublished method proposed by Geoff Hinton in
  http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
""" 
function RMSprop(; mu::Real = 0.9,
								 alpha::Real = 0.00000001, # TODO: investigate good defaults
								 beta::Real = 0.99, # TODO: investigate good defaults
								 epsilon::Real = 0.000001, # TODO: investigate good defaults
                                 alphaguess = LineSearches.InitialPrevious(), # TODO: investigate good defaults
                                 linesearch = LineSearches.HagerZhang(),        # TODO: investigate good defaults
                                 manifold::Manifold=Flat())
    RMSprop(mu,alpha, beta,epsilon, alphaguess, linesearch, manifold)
end

mutable struct RMSpropState{Tx, T} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
	y::Tx
    f_x_previous::T
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::RMSprop, options, d, initial_x)
    T = eltype(initial_x)
    initial_x = copy(initial_x)
	initial_y = copy(initial_x)
	fill!(initial_y,0.0)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)

    RMSpropState(initial_x, # Maintain current state in state.x
                                 copy(initial_x), # Maintain previous state in state.x_previous
								 initial_y,
                                 real(T)(NaN), # Store previous f in state.f_x_previous
                                 similar(initial_x), # Maintain current search direction in state.s
                                 @initial_linesearch()...)
end

function update_state!(d, state::RMSpropState, method::RMSprop)
    project_tangent!(method.manifold, gradient(d), state.x)
    # Search direction is always the negative gradient
    state.s .= .-gradient(d)
	
    # Update moving average of squared gradients
    state.y .= state.y.*method.beta .+(1-method.beta).*NaNMath.pow.(state.s,2)
	
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))
	
    # Update current position with alpha step + element calculated of RMSprop algorithm
    state.x .+= method.mu.*state.alpha.*state.s .+method.alpha.*state.s./(Base.sqrt.(state.y).+method.epsilon)
    retract!(method.manifold, state.x)
    lssuccess == false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::RMSprop, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end

function default_options(method::RMSprop)
    Dict(:allow_f_increases => true)
end
