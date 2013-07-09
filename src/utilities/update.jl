function update!(tr::OptimizationTrace,
                 iteration::Integer,
                 f_x::Real,
                 grnorm::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool)
    os = OptimizationState(iteration, f_x, grnorm, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        show(os)
    end
    return
end
