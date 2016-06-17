## Notes for contributing

If the change is more than just changing a typo, it will often be required to add
a bullet point to the `NEWS.md` file at the root of the repository. This makes it
easier for users to keep track of the changes since last version. A short description,
and a link to the PR or issue is sufficient.

### Adding a solver
If you're contributing a new solver, you shouldn't need to touch any of the code in
`src/optimize.jl`. You should rather add a file named (`solver` is the name of the solver)
`solver.jl` in `src`, and make sure that you define a trace macro `solvertrace`, an `Optimizer` subtype
`immutable Solver <: Optimizer end` with appropriate fields, a default constructor with a keyword
for each field, and an `optimize` method of the form

```
function optimize{T}(d::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::Solver,
                     o::OptimizationOptions)
    ...
end
```
