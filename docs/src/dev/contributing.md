## Notes for contributing

### Add a note in NEWS.md
If a change is more than just changing a typo, it will often be required to add
a bullet point to the `NEWS.md` file at the root of the repository. This makes it
easier for users to keep track of the changes since last version. A short description,
and a link to the PR or issue is sufficient.

### Adding a solver
If you're contributing a new solver, you shouldn't need to touch any of the code in
`src/optimize.jl`. You should rather add a file named (`solver` is the name of the solver)
`solver.jl` in `src`, and make sure that you define a trace macro `solvertrace`, an `Optimizer` subtype
`immutable Solver <: Optimizer end` with appropriate fields, a default constructor with a keyword
for each field, and an `optimize` method. Say you want to contribute a solver called
`Minim`, then your `src/minim.jl` file would look something like

```
macro minimtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
                dt["~inv(H)"] = copy(invH)
            end
            g_norm = vecnorm(g, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    g_norm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable Minim{T} <: Optimizer
    linesearch!::Function
    minim_parameter::T
end

Minim(; linesearch!::Function = hz_linesearch!, minim_parameter = 1.0) =
  Minim(linesearch!, minim_parameter)

function optimize{T}(d::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::Minim,
                     o::OptimizationOptions)
    ...
end
```
