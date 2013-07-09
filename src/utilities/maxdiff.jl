function maxdiff(x::Vector, y::Vector)
    res = 0.0
    for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
end
