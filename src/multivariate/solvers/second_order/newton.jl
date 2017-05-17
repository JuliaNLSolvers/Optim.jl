immutable Newton{L} <: Optimizer
    linesearch!::L
    resetalpha::Bool
end
#= uncomment v0.8.0
Newton(; linesearch::Function = LineSearches.HagerZhang()) =
Newton(linesearch)
=#
function Newton(; linesearch = LineSearches.HagerZhang(), resetalpha = true)
    Newton(linesearch,resetalpha)
end

Base.summary(::Newton) = "Newton's Method"

type NewtonState{T, N, F<:Base.LinAlg.Cholesky, Thd}
    x::Array{T,N}
    x_previous::Array{T, N}
    f_x_previous::T
    F::F
    Hd::Thd
    s::Array{T, N}
    @add_linesearch_fields()
end

function initial_state{T}(method::Newton, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    s = similar(initial_x)
    value_gradient!(d, initial_x)
    hessian!(d, initial_x)
    NewtonState(copy(initial_x), # Maintain current state in state.x
                similar(initial_x), # Maintain previous state in state.x_previous
                T(NaN), # Store previous f in state.f_x_previous
                Base.LinAlg.Cholesky(Matrix{T}(0, 0), :U),
                Vector{Int8}(),
                similar(initial_x), # Maintain current search direction in state.s
                @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::NewtonState{T}, method::Newton)
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.
    state.F, state.Hd = ldltfact!(Positive, NLSolversBase.hessian(d))
    state.s[:] = -(state.F\gradient(d))

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    lssuccess == false # break on linesearch error
end
