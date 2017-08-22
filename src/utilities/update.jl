function update!(tr::OptimizationTrace{T},
              iteration::Integer,
              f_x::Real,
              grnorm::Real,
              dt::Dict,
              store_trace::Bool,
              show_trace::Bool,
              show_every::Int = 1,
              callback = nothing) where T
    os = OptimizationState{T}(iteration, f_x, grnorm, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        if iteration % show_every == 0
            show(os)
        end
    end
    if callback != nothing && (iteration % show_every == 0)
        if store_trace
            stopped = callback(tr)
        else
            stopped = callback(os)
        end
    else
        stopped = false
    end
    stopped
end
