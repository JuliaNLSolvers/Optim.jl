
# Univariate Options
function optimize(f::F,
     lower::T,
     upper::T;
     method = Brent(),
     rel_tol::Real = sqrt(eps(T)),
     abs_tol::Real = eps(T),
     iterations::Integer = 1_000,
     store_trace::Bool = false,
     show_trace::Bool = false,
     callback = nothing,
     show_every = 1,
     extended_trace::Bool = false) where {F<:Function, T <: AbstractFloat}
    show_every = show_every > 0 ? show_every : 1
    if extended_trace && callback == nothing
        show_trace = true
    end

    show_trace && print_header(method)

    optimize(f, lower, upper, method;
             rel_tol = T(rel_tol),
             abs_tol = T(abs_tol),
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             show_every = show_every,
             callback = callback,
             extended_trace = extended_trace)
end

function optimize(f::F,
     lower::Real,
     upper::Real;
     kwargs...) where F<:Function
    optimize(f,
             Float64(lower),
             Float64(upper);
             kwargs...)
end

function optimize(f::F,
     lower::Real,
     upper::Real,
     mo::Union{Brent, GoldenSection};
     kwargs...) where F<:Function
    optimize(f,
             Float64(lower),
             Float64(upper),
             mo;
             kwargs...)
end
