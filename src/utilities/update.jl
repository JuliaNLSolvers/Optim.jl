function update!{T}(tr::OptimizationTrace{T},
                 iteration::Integer,
                 f_x::Real,
                 grnorm::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool,
                 show_every::Int = 1,
                 callback = nothing)
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

function ls_update!(out::AbstractArray, base::AbstractArray, step::AbstractArray, α)
    length(out) == length(base) == length(step) || throw(DimensionMismatch("all arrays must have the same length, got $(length(out)), $(length(base)), $(length(step))"))
    for i = 1:length(base)
        out[i] = base[i]+α*step[i]
    end
end
