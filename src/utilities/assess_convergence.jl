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

function assess_convergence(x::Vector,
                            x_previous::Vector,
                            f_x::Real,
                            f_x_previous::Real,
                            gr::Vector,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)
    x_converged, f_converged, gr_converged = false, false, false

    if maxdiff(x, x_previous) < xtol
        x_converged = true
    end
    if abs(f_x - f_x_previous) < ftol
        f_converged = true
    end
    if norm(gr, Inf) < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return x_converged, f_converged, gr_converged, converged
end
