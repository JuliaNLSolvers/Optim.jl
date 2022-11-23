
function print_header(method::Brent)
    @printf "Iter     Function value      Lower bound       Upper bound       Best bound\n"
end


function Base.show(io::IO, trace::OptimizationTrace{<:Real, Brent})
    @printf io "Iter     Function value      Lower bound       Upper bound       Best bound\n"
    @printf io "------   --------------      -----------       -----------       ----------\n"
    for state in trace.states
        show(io, state)
    end
    return
end

function Base.show(io::IO, t::OptimizationState{<:Real, Brent})
    @printf io "%6d   %14e    %14e    %14e      %s\n" t.iteration t.value t.metadata["x_lower"] t.metadata["x_upper"] t.metadata["best bound"]

    return
end

function print_header(method::GoldenSection)
    @printf "Iter     Function value      Lower bound       Upper bound\n"
end

function Base.show(io::IO, trace::OptimizationTrace{<:Real, GoldenSection})
    @printf io "Iter     Function value      Lower bound       Upper bound"
    @printf io "------   --------------      -----------       -----------"
    for state in trace.states
        show(io, state)
    end
    return
end

function Base.show(io::IO, t::OptimizationState{<:Real, GoldenSection})
    @printf io "%6d   %14e    %14e    %14e\n" t.iteration t.value t.metadata["x_lower"] t.metadata["x_upper"]

    return
end

function Base.show(io::IO, r::UnivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    @printf io " * Search Interval: [%f, %f]\n" lower_bound(r) upper_bound(r)
    @printf io " * Minimizer: %e\n" minimizer(r)
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" rel_tol(r) abs_tol(r) converged(r)
    @printf io " * Objective Function Calls: %d" f_calls(r)
    return
end
