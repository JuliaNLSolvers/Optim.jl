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
            xi >= li || return convert(T,Inf)
            phi += log(xi-li)
        end
        if isfinite(ui)
            xi <= ui || return convert(T,Inf)
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
            xi >= li || return convert(T,Inf)  # no need to compute additional gradient terms
            phi += log(xi-li)
            g[i] -= 1/(xi-li)
        end
        if isfinite(ui)
            xi <= ui || return convert(T,Inf)
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
            yi >= li || return convert(T,Inf)  # no need to compute additional gradient terms
            phi += log(yi-li)
        end
        if isfinite(ui)
            yi <= ui || return convert(T,Inf)
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

# combined_f(x, objective::Union{DifferentiableFunction,TwiceDifferentiableFunction}, constraints::AbstractConstraints, t) =
#     combined_f(x, objective.f, constraints, t)
#
# combined_g!(x, g, objective::Union{DifferentiableFunction,TwiceDifferentiableFunction}, constraints::AbstractConstraints, t) =
#     combined_g!(x, g, objective.g, constraints, t)
#
# combined_fg!(x, g, objective::Union{DifferentiableFunction,TwiceDifferentiableFunction}, constraints::AbstractConstraints, t) =
#     combined_fg!(x, g, objective.fg!, constraints, t)
#
# combined_h!(x, H, objective::Union{DifferentiableFunction,TwiceDifferentiableFunction}, constraints::AbstractConstraints, t) =
#     combined_h!(x, H, objective.h!, constraints, t)


#########################################
##
## Cost function for linear least squares
##
#########################################

linlsq(A::Matrix, b::Vector) =
    TwiceDifferentiableFunction( x    -> linlsq_fg!(x, Vector{eltype(x)}(0), A, b, similar(b)),
                                (x,g) -> linlsq_fg!(x, g, A, b, similar(b)),
                                (x,g) -> linlsq_fg!(x, g, A, b, similar(b)),
                                (x,H) -> linlsq_h!(x, H, A, b))

dummy_f(x)    = (Base.show_backtrace(STDOUT, backtrace()); error("No f"))
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
# Follows the notation of
# H. Hindi (2006), "A Tutorial on Convex Optimization II: Duality and
#     Interior Point Methods," Proc. 2006 American Control Conference.
# particularly his "Algorithm: Barrier method."

function interior_newton{T}(objective::TwiceDifferentiableFunction,
                            initial_x::Vector{T},
                            constraints::AbstractConstraints;
                            t = one(T), mu = 10, eps_gap = convert(T,NaN),
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
    H = Matrix{T}(m, m)
    lsr = LineSearchResults(T)

    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace

    df = DifferentiableFunction(dummy_f,
                                dummy_g!,
                                (x,gr) -> combined_fg!(x, gr, objective.fg!, constraints, t))
    iteration, iter_t, f_calls, g_calls = 0, 0, 0, 0
    f_x = f_x_previous = convert(T,Inf)
    x_converged = f_converged = gr_converged = converged = false
#    while m/t > eps_gap && iteration < iterations
    while iteration < iterations
        f_x_previous = f_x
        f_x = combined_fg!(x, gr, objective.fg!, constraints, t)
        combined_h!(x, H, objective.h!, constraints, t)
        F, Hd = ldltfact!(Positive, H)
        s = F\(-gr)
        sgr = vecdot(s, gr)
        if !(sgr <= 0) || isinf(sgr)
            # Presumably due to numeric instability
            converged = x_converged = f_converged = gr_converged = false
            break
        end
        clear!(lsr)
        push!(lsr, zero(T), f_x, sgr)
        f_x /= t
        @newtontrace
        alphamax = toedge(x, s, constraints)
        alpha = alphamax < 1 ? 0.9*alphamax : 1.0
        alpha, _f_calls, _g_calls =
            hz_linesearch!(df, x, s, xtmp, gr, lsr, alpha, true, constraints, alphamax)
        copy!(x_previous, x)
        step!(x, x, s, alpha, constraints)
        iteration += 1
        iter_t += 1
        f_calls += _f_calls
        g_calls += _g_calls
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
                                       t*grtol)
        # It's time to change the barrier penalty if we're
        # sufficiently "in the basin." That means (1) the Hessian is
        # positive (semi)definite (all(Hd .> 0)) and that the Hessian
        # step is accepted (alpha is approximately 1).  The latter
        # suggests that the Hessian is a reliable predictor of the
        # function shape, and that within a few iterations we'd be at
        # the minimum.
        if (alpha >= 0.9 && all(Hd .>= 0)) || converged
            if iter_t == 1 && converged
                # if changing t didn't change the solution, quit now
                break
            end
            t *= mu
            iter_t = 0
            df = DifferentiableFunction(dummy_f,
                                        dummy_g!,
                                        (x,gr) -> combined_fg!(x, gr, objective.fg!, constraints, t))
        end
    end
    MultivariateOptimizationResults("Interior/Newton",
                                    initial_x,
                                    x,
                                    @compat(Float64(f_x)),
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


function interior{T}(objective::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                     initial_x::Array{T},
                     constraints::AbstractConstraints;
                     method = :cg, t = convert(T,NaN), mu = 10, eps_gap = convert(T,NaN),
                     xtol::Real = 1e-32,
                     ftol::Real = 1e-8,
                     grtol::Real = 1e-8,
                     iterations::Integer = 1_000,
                     store_trace::Bool = false,
                     show_trace::Bool = false,
                     extended_trace::Bool = false)
    if isnan(t)
        vo, vc, gogo, gogc = gnorm(objective, constraints, initial_x)
        if isfinite(vo) && isfinite(vc) && sqrt(gogo) < grtol
            return MultivariateOptimizationResults("Interior",
                                    initial_x,
                                    initial_x,
                                    Float64(vo),
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
        if gogc == 0
            t = one(T)  # FIXME: bounds constraints, starting at exact center. This is probably not the right guess.
        else
            t = convert(T, 10*abs(gogc)/gogo)
        end
    end
    if isnan(eps_gap)
        eps_gap = (cbrt(eps(T)))^2/t
    end
    if method == :newton
        return interior_newton(objective, initial_x, constraints; t=t, mu=mu, eps_gap=eps_gap, xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations, store_trace=store_trace, show_trace=show_trace, extended_trace=extended_trace)
    end
    if !feasible(initial_x, constraints)
        error("Initial guess must be feasible")
    end
    x = copy(initial_x)
    m = length(x)
    local results
    iteration, f_calls, g_calls = 0, 0, 0
    while m/t > eps_gap || iteration == 0
        df = DifferentiableFunction( x    ->   combined_f(x, objective.f, constraints, t),
                                    (x,g) ->  combined_g!(x, g, objective.g!, constraints, t),
                                    (x,g) -> combined_fg!(x, g, objective.fg!, constraints, t))
        results = optimize(df, x, method=method, constraints=constraints, interior=true, xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations, store_trace=store_trace, show_trace=show_trace, extended_trace=extended_trace)
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

"""
`vo, vc, gogo, gogc = gnorm(objective, constraints, initial_x)`
computes the values `vo`, `vc` of the `objective` and `constraints` at
position `initial_x`, as well as their gradients. `gogo` is the dot
product of the objective gradient with itself; `gogc` is the dot
product of the objective gradient and the constraint gradient.

The latter two can be used to initialize the tradeoff between
objective and constraints so that the total gradient `t*go + gc` is
still a descent direction for the objective. This ensures that the
barrier penalty will not push the solution out of the starting basin.
"""
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
