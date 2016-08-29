# TODO decide if this is wanted and/or necessary
macro add_generic_fields()
    quote
        method_string::String
        n::Int64
        x::Array{T}
        f_x::T
        f_calls::Int64
        g_calls::Int64
        h_calls::Int64
        elapsed::Float64
    end
end

macro add_linesearch_fields()
    quote
        x_ls::Array{T}
        g_ls::Array{T}
        alpha::T
        mayterminate::Bool
        lsr::LineSearchResults
    end
end

macro initial_linesearch()
    quote
        (similar(initial_x), # Buffer of x for line search in state.x_ls
        similar(initial_x), # Buffer of g for line search in state.g_ls
        alphainit(one(T), initial_x, g, f_x), # Keep track of step size in state.alpha
        false, # state.mayterminate
        LineSearchResults(T))
    end
end
