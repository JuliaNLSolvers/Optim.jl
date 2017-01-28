immutable Newton <: Optimizer
    linesearch!::Function
    resetalpha::Bool
end
#= uncomment v0.8.0
Newton(; linesearch::Function = LineSearches.hagerzhang!) =
Newton(linesearch)
=#
function Newton(; linesearch! = nothing, linesearch::Function = LineSearches.hagerzhang!,
                resetalpha = true)
    linesearch = get_linesearch(linesearch!, linesearch)
    Newton(linesearch,resetalpha)
end

type NewtonState{T, N, F<:Base.LinAlg.Cholesky, Thd}
    @add_generic_fields()
    x_previous::Array{T, N}
    g::Array{T, N}
    f_x_previous::T
    H::Matrix{T}
    F::F
    Hd::Thd
    s::Array{T, N}
    @add_linesearch_fields()
end

function initial_state{T}(method::Newton, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    g = similar(initial_x)
    s = similar(g)
    x_ls, g_ls = similar(g), similar(g)
    f_x_previous, f_x = NaN, d.fg!(initial_x, g)
    f_calls, g_calls = 1, 1
    H = Array{T}(n, n)
    d.h!(initial_x, H)
    h_calls = 1

    NewtonState("Newton's Method",
                length(initial_x),
                copy(initial_x), # Maintain current state in state.x
                f_x, # Store current f in state.f_x
                f_calls, # Track f calls in state.f_calls
                g_calls, # Track g calls in state.g_calls
                h_calls,
                similar(initial_x), # Maintain previous state in state.x_previous
                g, # Store current gradient in state.g
                T(NaN), # Store previous f in state.f_x_previous
                H,
                Base.LinAlg.Cholesky(Matrix{T}(), :U),
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
    state.F, state.Hd = ldltfact!(Positive, state.H)
    state.s[:] = -(state.F\state.g)

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    lssuccess == false # break on linesearch error
end
