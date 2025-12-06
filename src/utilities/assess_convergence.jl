f_abschange(state::AbstractOptimizerState) = f_abschange(state.f_x, state.f_x_previous)
f_abschange(f_x, f_x_previous) = abs(f_x - f_x_previous)
f_relchange(state::AbstractOptimizerState) = f_relchange(state.f_x, state.f_x_previous)
f_relchange(f_x, f_x_previous) = abs(f_x - f_x_previous) / abs(f_x)

x_abschange(state::AbstractOptimizerState) = x_abschange(state.x, state.x_previous)
x_abschange(x::AbstractArray{<:Number}, x_previous::AbstractArray{<:Number}) = Linfdist(x, x_previous)
x_relchange(state::AbstractOptimizerState) = x_relchange(state.x, state.x_previous)
x_relchange(x::AbstractArray{<:Number}, x_previous::AbstractArray{<:Number}) = Linfdist(x, x_previous) / Base.maximum(abs, x) # Base.maximum !== maximum

# Copied and adapted from https://github.com/JuliaStats/StatsBase.jl/blob/d70c4a203177c79d851c1cdca450bc6dbd2a4683/src/deviation.jl#L100-L110
function Linfdist(x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    n = length(x)
    length(y) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    if iszero(n)
        return zero(abs(zero(eltype(x)) - zero(eltype(y))))
    else
        broadcasted = Broadcast.broadcasted((xi, yi) -> abs(xi - yi), vec(x), vec(y))
        return Base.maximum(Broadcast.instantiate(broadcasted)) # Base.maximum !== maximum
    end
end

g_residual(state::NelderMeadState) = state.nm_x
g_residual(state::ZerothOrderState) = oftype(state.f_x, NaN)
g_residual(state::AbstractOptimizerState) = g_residual(state.g_x)
g_residual(g_x::AbstractArray) = Base.maximum(abs, g_x) # Base.maximum !== maximum
gradient_convergence_assessment(state::AbstractOptimizerState, options::Options) =
    g_residual(state) ≤ options.g_abstol
gradient_convergence_assessment(state::ZerothOrderState, options::Options) = false

# Default function for convergence assessment used by
# AcceleratedGradientDescentState, BFGSState, ConjugateGradientState,
# GradientDescentState, LBFGSState, MomentumGradientDescentState and NewtonState
function assess_convergence(state::AbstractOptimizerState, d, options::Options)
    assess_convergence(
        state.x,
        state.x_previous,
        state.f_x,
        state.f_x_previous,
        state.g_x,
        options.x_abstol,
        options.x_reltol,
        options.f_abstol,
        options.f_reltol,
        options.g_abstol,
    )
end
function assess_convergence(
    x,
    x_previous,
    f_x,
    f_x_previous,
    g_x,
    x_abstol,
    x_reltol,
    f_abstol,
    f_reltol,
    g_abstol,
)
    # TODO: Create function for x_convergence_assessment
    x_converged = x_abschange(x, x_previous) ≤ x_abstol ||
        x_abschange(x, x_previous) ≤ x_reltol * Base.maximum(abs, x) # Base.maximum !== maximum

    # Relative Tolerance
    # TODO: Create function for f_convergence_assessment
    f_converged = f_abschange(f_x, f_x_previous) ≤ f_abstol ||
        f_abschange(f_x, f_x_previous) ≤ f_reltol * abs(f_x)

    f_increased = f_x > f_x_previous

    g_converged = g_residual(g_x) ≤ g_abstol # Base.maximum !== maximum

    return x_converged, f_converged, g_converged, f_increased
end

# Used by Fminbox and IPNewton
function assess_convergence(x, x_previous, f_x, f_x_previous, g_x, x_tol, f_tol, g_tol)
    x_converged = x_abschange(x, x_previous) ≤ x_tol

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    f_converged = f_abschange(f_x, f_x_previous) ≤ f_tol * abs(f_x)

    f_increased = f_x > f_x_previous

    g_converged = g_residual(g_x) ≤ g_tol # Base.maximum !== maximum

    return x_converged, f_converged, g_converged, f_increased
end
