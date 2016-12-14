## Notes for contributing

### Add a note in NEWS.md
If a change is more than just changing a typo, it will often be required to add
a bullet point to the `NEWS.md` file at the root of the repository. This makes it
easier for users to keep track of the changes since last version. A short description,
and a link to the PR or issue is sufficient.

### Adding a solver
If you're contributing a new solver, you shouldn't need to touch any of the code in
`src/optimize.jl`. You should rather add a file named (`solver` is the name of the solver)
`solver.jl` in `src`, and make sure that you define an `Optimizer` subtype
`immutable Solver <: Optimizer end` with appropriate fields, a default constructor with a keyword
for each field, a state type that holds all variables that are (re)used throughout
the iterative procedure, an `initial_state` that initializes such a state, and  an `update!` method
that does the actual work. Say you want to contribute a solver called
`Minim`, then your `src/minim.jl` file would look something like

```
immutable Minim{F<:Function, T} <: Optimizer
    linesearch!::F
    minim_parameter::T
end

Minim(; linesearch! = LineSearches.hagerzhang!, minim_parameter = 1.0) =
  Minim(linesearch!, minim_parameter)

type MinimState{T}
  @add_generic_fields()
  x_previous::Array{T}
  g::Array{T}
  f_x_previous::T
  s::Array{T}
  @add_linesearch_fields()
end

function initial_state(method::Minim, options, d, initial_x)
# prepare cache variables etc here

end

function update!{T}(d, state::MinimState{T}, method::Minim)
    # code for Minim here
    false # should the procedure force quit?
end
```
