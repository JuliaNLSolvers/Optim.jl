macro def(name, definition)
  esc(quote
    macro $name()
      esc($(Expr(:quote, definition)))
    end
  end)
end

@def add_linesearch_fields begin
    dphi_0_previous::T
    x_ls::Tx
    alpha::T
end

@def initial_linesearch begin
    (T(NaN),            # Keep track of previous descent value ⟨∇f(x_{k-1}), s_{k-1}⟩
    similar(initial_x), # Buffer of x for line search in state.x_ls
    one(T))             # Keep track of step size in state.alpha
end
