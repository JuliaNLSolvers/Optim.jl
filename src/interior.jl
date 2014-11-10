############
##
##  Barriers
##
############
# Note: barrier_g! and barrier_h! simply add to an existing g or H
# (they do not initialize with zeros)

## Box constraints
function barrier_f{T}(x::Array{T}, bounds::ConstraintsBox)
    n = length(x)
    phi = zero(T)
    l = bounds.lower
    u = bounds.upper
    for i = 1:n
        li, ui = l[i], u[i]
        xi = x[i]
        if isfinite(li)
            xi >= li || return inf(T)
            phi += log(xi-li)
        end
        if isfinite(ui)
            xi <= ui || return inf(T)
            phi += log(ui-xi)
        end
    end
    -phi::T
end

function barrier_g!{T}(x::Array{T}, g::Array{T}, bounds::ConstraintsBox)
    n = length(x)
    l = bounds.lower
    u = bounds.upper
    for i = 1:n
        li, ui = l[i], u[i]
        xi = x[i]
        g[i] -= 1/(xi-li)
        g[i] += 1/(ui-xi)
    end
end

function barrier_fg!{T}(x::Array{T}, g::Array{T}, bounds::ConstraintsBox)
    n = length(x)
    phi = zero(T)
    l = bounds.lower
    u = bounds.upper
    for i = 1:n
        li, ui = l[i], u[i]
        xi = x[i]
        if isfinite(li)
            xi >= li || return inf(T)  # no need to compute additional gradient terms
            phi += log(xi-li)
            g[i] -= 1/(xi-li)
        end
        if isfinite(ui)
            xi <= ui || return inf(T)
            phi += log(ui-xi)
            g[i] += 1/(ui-xi)
        end
    end
    -phi::T
end

function barrier_h!{T}(x::Vector{T}, H::Matrix{T}, bounds::ConstraintsBox)
    n = length(x)
    l = bounds.lower
    u = bounds.upper
    for i = 1:n
        li, ui = l[i], u[i]
        xi = x[i]
        dx = xi-li
        H[i,i] += 1/(dx*dx)
        dx = ui-xi
        H[i,i] += 1/(dx*dx)
    end
end

## Linear constraints
function barrier_fg!{T}(x::Array{T}, g::Array{T}, constraints::ConstraintsL)
    m, n = size(constraints.A)
    y1 = constraints.scratch1
    A_mul_B!(y1, constraints.A, x)
    phi = zero(T)
    l = constraints.lower
    u = constraints.upper
    lenx = length(x)
    lenc = length(y1)
    for i = 1:lenc
        li, ui = l[i], u[i]
        yi = y1[i]
        if isfinite(li)
            yi >= li || return inf(T)  # no need to compute additional gradient terms
            phi += log(yi-li)
        end
        if isfinite(ui)
            yi <= ui || return inf(T)
            phi += log(ui-yi)
        end
    end
    if isempty(g)
        return -phi
    end
    y2 = constraints.scratch2
    y3 = constraints.scratch3
    for i = 1:lenc
        y2[i] = 1/(y1[i] - l[i])
    end
    At_mul_B!(y3, constraints.A, y2)
    for i = 1:lenx
        g[i] -= y3[i]
    end
    for i = 1:lenc
        y2[i] = 1/(u[i] - y1[i])
    end
    At_mul_B!(y3, constraints.A, y2)
    for i = 1:lenx
        g[i] += y3[i]
    end
    -phi::T
end

barrier_f{T}(x::Array{T}, constraints::ConstraintsL) = barrier_fg!(x, T[], constraints)


## Nonlinear inequalities


#########################
##
## Combined cost function
##
#########################
combined_f(x, objective_f::Function, constraints::AbstractConstraints, t) =
    t*objective_f(x) + barrier_f(x, constraints)

function combined_g!(x, g, objective_g!::Function, constraints::AbstractConstraints, t)
    objective_g!(x, g)
    scale!(g, t)
    barrier_fg!(x, g, constraints)
end

function combined_fg!(x, g, objective_fg!::Function, constraints::AbstractConstraints, t)
    f_x = t*objective_fg!(x, g)
    scale!(g, t)
    f_x += barrier_fg!(x, g, constraints)
    f_x
end

function combined_h!(x, H, objective_h!::Function, constraints::AbstractConstraints, t)
    objective_h!(x, H)
    scale!(H, t)
    barrier_h!(x, H, constraints)
end

# combined_f(x, objective::Union(DifferentiableFunction,TwiceDifferentiableFunction), constraints::AbstractConstraints, t) =
#     combined_f(x, objective.f, constraints, t)
# 
# combined_g!(x, g, objective::Union(DifferentiableFunction,TwiceDifferentiableFunction), constraints::AbstractConstraints, t) =
#     combined_g!(x, g, objective.g, constraints, t)
# 
# combined_fg!(x, g, objective::Union(DifferentiableFunction,TwiceDifferentiableFunction), constraints::AbstractConstraints, t) =
#     combined_fg!(x, g, objective.fg!, constraints, t)
# 
# combined_h!(x, H, objective::Union(DifferentiableFunction,TwiceDifferentiableFunction), constraints::AbstractConstraints, t) =
#     combined_h!(x, H, objective.h!, constraints, t)


#########################################
##
## Cost function for linear least squares
##
#########################################
    
linlsq(A::Matrix, b::Vector) =
    TwiceDifferentiableFunction( x    -> linlsq_fg!(x, Array(eltype(x),0), A, b, similar(b)),
                                (x,g) -> linlsq_fg!(x, g, A, b, similar(b)),
                                (x,g) -> linlsq_fg!(x, g, A, b, similar(b)),
                                (x,H) -> linlsq_h!(x, H, A, b))

dummy_f (x)   = (Base.show_backtrace(STDOUT, backtrace()); error("No f"))
dummy_g!(x,g) = (Base.show_backtrace(STDOUT, backtrace()); error("No g!"))

function linlsq_fg!{T}(x::AbstractArray{T}, g, A, b, scratch)
    A_mul_B!(scratch, A, x)
    f_x = zero(typeof(one(T)*one(T)+one(T)*one(T)))
    for i = 1:length(g)
        tmp = scratch[i]
        tmp -= b[i]
        f_x += tmp*tmp
        scratch[i] = tmp
    end
    if !isempty(g)
        At_mul_B!(g, A, scratch)
    end
    f_x/2
end

linlsq_h!(x, H, A, b) = At_mul_B!(H, A, A)


#########################
##
## Interior point methods
##
#########################
function interior_newton{T}(objective::TwiceDifferentiableFunction,
                            initial_x::Vector{T},
                            constraints::AbstractConstraints;
                            t = one(T), mu = 10, eps_gap = 1e-12,
                            xtol::Real = convert(T,1e-32),
                            ftol::Real = convert(T,1e-8),
                            grtol::Real = convert(T,1e-8),
                            iterations::Integer = 1_000,
                            store_trace::Bool = false,
                            show_trace::Bool = false,
                            extended_trace::Bool = false)

    if !feasible(initial_x, constraints)
        error("Initial guess must be feasible")
    end
    x = copy(initial_x)
    xtmp = similar(x)
    x_previous = similar(x)
    m = length(x)
    gr = similar(x)
    H = Array(T, m, m)
    lsr = LineSearchResults(T)

    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    
    df = DifferentiableFunction(dummy_f,
                                dummy_g!,
                                (x,gr) -> combined_fg!(x, gr, objective.fg!, constraints, t))
    iteration, f_calls, g_calls = 0, 0, 0
    f_x = f_x_previous = inf(T)
    while m/t > eps_gap && iteration < iterations
        f_x_previous = f_x
        f_x = combined_fg!(x, gr, objective.fg!, constraints, t)
        combined_h!(x, H, objective.h!, constraints, t)
        @newtontrace
        cH = cholfact!(H)
        s = -(cH\gr)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dot(s,gr))
        alphamax = toedge(x, s, constraints)
        alpha = alphamax < 1 ? 0.9*alphamax : 1.0
        alpha, _f_calls, _g_calls =
            hz_linesearch!(df, x, s, xtmp, gr, lsr, alpha, true, constraints, alphamax)
        copy!(x_previous, x)
        step!(x, x, s, alpha, constraints)
        iteration += 1
        f_calls += _f_calls
        g_calls += _g_calls
        f_x /= t
        if alphamax >= 1
            t *= mu
            df = DifferentiableFunction(dummy_f,
                                        dummy_g!,
                                        (x,gr) -> combined_fg!(x, gr, objective.fg!, constraints, t))
        end
    end
    x_converged,
    f_converged,
    gr_converged,
    converged = assess_convergence(x,
                                   x_previous,
                                   f_x,
                                   f_x_previous,
                                   gr,
                                   xtol,
                                   ftol,
                                   grtol)
    MultivariateOptimizationResults("Interior/Newton",
                                    initial_x,
                                    x,
                                    float64(f_x),
                                    iteration,
                                    iteration == iterations,
                                    x_converged,
                                    xtol,
                                    f_converged,
                                    ftol,
                                    gr_converged,
                                    grtol,
                                    tr,
                                    f_calls,
                                    g_calls)
end


function interior{T}(objective::Union(DifferentiableFunction, TwiceDifferentiableFunction),
                     initial_x::Array{T},
                     constraints::AbstractConstraints;
                     method = :cg, t = nan(T), mu = 10, eps_gap = 1e-12,
                     xtol::Real = 1e-32,
                     ftol::Real = 1e-8,
                     grtol::Real = 1e-8,
                     iterations::Integer = 1_000,
                     store_trace::Bool = false,
                     show_trace::Bool = false,
                     extended_trace::Bool = false)
    if isnan(t)
        vo, vc, go, gc = gnorm(objective, constraints, initial_x)
        if isfinite(vo) && isfinite(vc) && go < grtol
            return MultivariateOptimizationResults("Interior",
                                    initial_x,
                                    initial_x,
                                    float64(vo),
                                    0,
                                    false,
                                    false,
                                    xtol,
                                    false,
                                    ftol,
                                    true,
                                    grtol,
                                    OptimizationTrace(),
                                    0,
                                    0)
        end
        if gc == 0
            t = one(T)  # FIXME: bounds constraints, starting at exact center. This is probably not the right guess.
        else
            t = gc/(convert(T,0.1)*go)
        end
    end
    if method == :newton
        return iterior_newton(objective, initial_x, constraints; t=t, mu=mu, eps_gap=eps_gap) # FIXME
    end
    if !feasible(initial_x, constraints)
        error("Initial guess must be feasible")
    end
    x = copy(initial_x)
    m = length(x)
    local results
    iteration, f_calls, g_calls = 0, 0, 0
    while m/t > eps_gap || iteration == 0
        df = DifferentiableFunction( x    -> combined_f  (x, objective.f, constraints, t),
                                    (x,g) -> combined_g! (x, g, objective.g!, constraints, t),
                                    (x,g) -> combined_fg!(x, g, objective.fg!, constraints, t))
        results = optimize(df, x, method=method, xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations, store_trace=store_trace, show_trace=show_trace, extended_trace=extended_trace)
        copy!(x, results.minimum)
        iteration += results.iterations
        f_calls += results.f_calls
        g_calls += results.g_calls
        results.f_minimum /= t
        t *= mu
    end
    results.iterations, results.f_calls, results.g_calls = iteration, f_calls, g_calls
    copy!(results.initial_x, initial_x)
    results
end

function gnorm(objective, constraints, initial_x)
    go = similar(initial_x)
    gc = similar(initial_x)
    fill!(gc, 0)
    vo = objective.fg!(initial_x, go)
    vc = barrier_fg!(initial_x, gc, constraints)
    gom = gcm = zero(eltype(go))
    for i = 1:length(go)
        gom += abs(go[i])
        gcm += abs(gc[i])
    end
    vo, vc, gom, gcm
end
