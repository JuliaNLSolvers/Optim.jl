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
