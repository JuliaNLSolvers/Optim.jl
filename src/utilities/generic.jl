macro def(name, definition)
    esc(quote
        macro $name()
            esc($(Expr(:quote, definition)))
        end
    end)
end

@def add_linesearch_fields begin
    x_ls::Tx
    alpha::T
end

@def initial_linesearch begin
    (
        fill!(similar(x0), NaN), # Buffer of x for line search in state.x_ls
        real(oneunit(eltype(x0))),
    )             # Keep track of step size in state.alpha
end
