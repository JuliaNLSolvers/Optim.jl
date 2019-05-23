function print_header(method::IPOptimizer)
    @printf "Iter     Lagrangian value Function value   Gradient norm    |==constr.|      μ\n"
end

function Base.show(io::IO, t::OptimizationState{Tf, M}) where M<:IPOptimizer where Tf
    md = t.metadata
    @printf io "%6d   %-14e   %-14e   %-14e   %-14e   %-6.2e\n" t.iteration md["Lagrangian"] t.value t.g_norm md["ev"] md["μ"]
    if !isempty(t.metadata)
        for (key, value) in md
            key ∈ ("Lagrangian", "μ", "ev") && continue
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace{Tf, M}) where M <: IPOptimizer where Tf
    @printf io "Iter     Lagrangian value Function value   Gradient norm    |==constr.|      μ\n"
    @printf io "------   ---------------- --------------   --------------   --------------   --------\n"
    for state in tr
        show(io, state)
    end
    return
end

function trace!(tr, d, state, iteration, method::IPOptimizer, options, curr_time=time())
    dt = Dict()
    dt["Lagrangian"] = state.L
    dt["μ"] = state.μ
    dt["ev"] = abs(state.ev)
    dt["time"] = curr_time
    if options.extended_trace
        dt["α"] = state.alpha
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["h(x)"] = copy(state.H)
        if !isempty(state.bstate)
            dt["gtilde(x)"] = copy(state.gtilde)
            dt["bstate"] = copy(state.bstate)
            dt["bgrad"] = copy(state.bgrad)
            dt["c"] = copy(state.constr_c)
        end
    end
    g_norm = norm(state.g, Inf) + norm(state.bgrad, Inf)
    Optim.update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
