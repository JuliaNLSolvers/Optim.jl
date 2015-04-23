# centroid except h-th vertex
function centroid!(c, simplex, h=0)
    n = length(c)
    fill!(c, 0.0)
    @inbounds for i in 1:n+1
        if i != h
            xi = simplex[i]
            for j in 1:n
                c[j] += xi[j]
            end
        end
    end
    for j in 1:n
        c[j] /= n
    end
    c
end

centroid(simplex, h=0) = centroid!(similar(simplex[1]), simplex, h)

macro nmtrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(xl)
            end
            grnorm = NaN
            update!(tr,
                    iteration,
                    fl,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

macro fcall(x)
    quote
        f_calls += 1
        f($x)
    end
end

# References:
# * Fuchang Gao and Lixing Han (2010), Springer US. "Implementing the Nelder-Mead simplex algorithm with adaptive parameters" (doi:10.1007/s10589-010-9329-3)
# * Saša Singer and John Nelder (2009), Scholarpedia, 4(7):2928. "Nelder-Mead algorithm" (doi:10.4249/scholarpedia.2928)

function nelder_mead{T}(f::Function,
                        initial_x::Vector{T};
                        ftol::Real = 1e-8,
                        xtol::Real = 1e-8,
                        initial_step::Vector{T} = ones(T,length(initial_x)),
                        iterations::Integer = 1_000,
                        store_trace::Bool = false,
                        show_trace::Bool = false,
                        extended_trace::Bool = false)
    n = length(initial_x)
    if n == 1
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end

    # parameters of transformations
    α = 1.0
    β = 1.0 + 2 / n
    γ = 0.75 - 1 / 2n
    δ = 1.0 - 1 / n

    # NOTE: Use @fcall macro to evaluate a function values
    #       so as to automatically increment a function call counter
    # number of function calls
    f_calls = 0

    # initialize a simplex and function values
    simplex = Vector{Float64}[copy(initial_x)]
    fvalues = Float64[@fcall(initial_x)]
    for i in 1:n
        x = similar(initial_x)
        τ = initial_step[i]
        @inbounds for j in 1:n
            x[j] = initial_x[j] + ifelse(j == i, τ, 0.0)
        end
        push!(simplex, x)
        push!(fvalues, @fcall(x))
    end
    ord = sortperm(fvalues)

    iteration = 0

    # centroid cache
    c = centroid(simplex, ord[n+1])

    # transformed points
    xr = similar(initial_x)
    xe = similar(initial_x)
    xc = similar(initial_x)

    # Maintain a trace
    tr = OptimizationTrace()
    tracing = show_trace || store_trace || extended_trace
    l = ord[1]
    xl = simplex[l]
    fl = fvalues[l]
    @nmtrace

    # Iterate until convergence or exhaustion
    x_converged = false
    f_converged = false

    while !(x_converged && f_converged) && iteration < iterations
        # Augment the iteration counter
        iteration += 1

        # assertions for debug
        # * function values (`fvalues`) should be sorted according to the `ord` indices
        #@assert issorted(fvalues[ord])
        # * centroid cache (`c`) should be close enough to the centroid of the `simplex` (excluding its highest vertex)
        #@assert norm(c .- centroid(simplex, ord[n+1]), Inf) < xtol * 1.0e-2

        # highest, second highest, and lowest indices, respectively
        h = ord[n+1]
        s = ord[n]
        l = ord[1]

        xh = simplex[h]
        fh = fvalues[h]
        fs = fvalues[s]
        xl = simplex[l]
        fl = fvalues[l]

        # reflect
        @inbounds for j in 1:n
            xr[j] = c[j] + α * (c[j] - xh[j])
        end
        fr = @fcall xr
        doshrink = false

        if fr < fl # <= fs
            # expand
            @inbounds for j in 1:n
                xe[j] = c[j] + β * (xr[j] - c[j])
            end
            fe = @fcall xe
            if fe < fr
                accept = (xe, fe)
            else
                accept = (xr, fr)
            end
        elseif fr < fs
            accept = (xr, fr)
        else # fs <= fr
            # contract
            if fr < fh
                # outside
                @inbounds for j in 1:n
                    xc[j] = c[j] + γ * (xr[j] - c[j])
                end
                fc = @fcall xc
                if fc <= fr
                    accept = (xc, fc)
                else
                    doshrink = true
                end
            else
                # inside
                @inbounds for j in 1:n
                    xc[j] = c[j] - γ * (xr[j] - c[j])
                end
                fc = @fcall xc
                if fc < fh
                    accept = (xc, fc)
                else
                    doshrink = true
                end
            end

            # shrinkage almost never happen in practice
            if doshrink
                # shrink
                for i in 2:n+1
                    o = ord[i]
                    xi = xl .+ δ * (simplex[o] .- xl)
                    simplex[o] = xi
                    fvalues[o] = @fcall xi
                end
            end
        end

        if doshrink
            # TODO: use in-place sortperm (v0.4 has it!)
            ord = sortperm(fvalues)
            centroid!(c, simplex, ord[n+1])
        else
            x, fvalue = accept
            @inbounds for j in 1:n
                simplex[h][j] = x[j]
            end

            # insert the new function value into an ordered position
            fvalues[h] = fvalue
            @inbounds for i in n+1:-1:2
                if fvalues[ord[i-1]] > fvalues[ord[i]]
                    ord[i-1], ord[i] = ord[i], ord[i-1]
                else
                    break
                end
            end

            # add the new vertex, and subtract the highest vertex
            h = ord[n+1]
            xh = simplex[h]
            @inbounds for j in 1:n
                c[j] += (x[j] - xh[j]) / n
            end
        end

        l = ord[1]
        xl = simplex[l]
        fl = fvalues[l]
        @nmtrace

        # check convergence
        x_converged = true
        @inbounds for i in 1:n+1, j in 1:n
            if abs(simplex[i][j] - xl[j]) > xtol
                x_converged = false
                break
            end
        end
        f_converged = true
        @inbounds for i in 1:n+1
            if abs(fvalues[i] - fl) > ftol
                f_converged = false
                break
            end
        end
    end

    centroid!(c, simplex)
    fcent = @fcall c
    l = ord[1]
    if fcent < fvalues[l]
        x = c
        f_x = fcent
    else
        x = simplex[l]
        f_x = fvalues[l]
    end

    return MultivariateOptimizationResults("Nelder-Mead",
                                           initial_x,
                                           convert(typeof(initial_x), x),
                                           float64(f_x),
                                           iteration,
                                           iteration == iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           false,
                                           NaN,
                                           tr,
                                           f_calls,
                                           0)
end
