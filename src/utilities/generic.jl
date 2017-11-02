macro def(name, definition)
  esc(quote
    macro $name()
      esc($(Expr(:quote, definition)))
    end
  end)
end

@def add_linesearch_fields begin
    x_ls::Array{T,N}
    alpha::T
    mayterminate::Bool
    lsr::LineSearches.LineSearchResults
end

@def initial_linesearch begin
    (similar(initial_x), # Buffer of x for line search in state.x_ls
    one(T), # Keep track of step size in state.alpha
    false, # state.mayterminate
    LineSearches.LineSearchResults(T))
end
