function ls_update!(out::AbstractArray, base::AbstractArray, step::AbstractArray, α)
    length(out) == length(base) == length(step) || throw(DimensionMismatch("all arrays must have the same length, got $(length(out)), $(length(base)), $(length(step))"))
    @. out  = base + α*step
end
