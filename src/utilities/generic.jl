# TODO decide if this is wanted and/or necessary
macro add_generic_fields()
    quote
        x::Array
        f_x::Float64
        f_calls::Int64
        g_calls::Int64
        h_calls::Int64
        elapsed::Float64
    end
end
