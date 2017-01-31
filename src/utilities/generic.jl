macro def(name, definition)
  esc(quote
    macro $name()
      esc($(Expr(:quote, definition)))
    end
  end)
end

# TODO decide if this is wanted and/or necessary
@def add_generic_fields begin
    method_string::String
    n::Int
    x::Array{T,N}
end

@def add_linesearch_fields begin
    x_ls::Array{T,N}
    g_ls::Array{T,N}
    alpha::T
    mayterminate::Bool
    lsr::LineSearches.LineSearchResults
end

@def initial_linesearch begin
    (similar(initial_x), # Buffer of x for line search in state.x_ls
    similar(initial_x), # Buffer of g for line search in state.g_ls
    LineSearches.alphainit(one(T), initial_x, gradient(d), value(d)), # Keep track of step size in state.alpha
    false, # state.mayterminate
    LineSearches.LineSearchResults(T))
end
