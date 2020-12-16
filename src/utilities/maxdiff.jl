function maxdiff(x::AbstractArray, y::AbstractArray)
    return mapreduce((a, b) -> abs(a - b), max, x, y)
end
