# Univariate Options
function optimize(f,
     lower::T,
     upper::T;
     method = Brent(),
     rel_tol::Real = sqrt(eps(float(T))),
     abs_tol::Real = eps(float(T)),
     iterations::Integer = 1_000,
     store_trace::Bool = false,
     show_trace::Bool = false,
     callback = nothing,
     show_every = 1,
     extended_trace::Bool = false) where T <: Real
    show_every = show_every > 0 ? show_every : 1
    if extended_trace && callback === nothing
        show_trace = true
    end

    show_trace && print_header(method)
    Tf = float(T)
    optimize(f, Tf(lower), Tf(upper), method;
             rel_tol = Tf(rel_tol),
             abs_tol = Tf(abs_tol),
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             show_every = show_every,
             callback = callback,
             extended_trace = extended_trace)
end

function optimize(f,
    lower::Union{Integer, Real},
    upper::Union{Integer, Real};
    kwargs...)
     
    T = promote_type(typeof(lower/1), typeof(upper/1))
    optimize(f,
             T(lower),
             T(upper);
             kwargs...)
end

function optimize(f,
    lower::Union{Integer, Real},
    upper::Union{Integer, Real},
    method::Union{Brent, GoldenSection};
    kwargs...)
     
    T = promote_type(typeof(lower/1), typeof(upper/1))
    optimize(f,
             T(lower),
             T(upper),
             method;
             kwargs...)
end
