## Notes for contributing
We are always happy to get help from people who normally do not contribute to the package. However, to make the process run smoothly, we ask you to read this page before creating your pull request. That way it is more probable that your changes will be incorporated, and in the end it will mean less work for everyone.

### Things to consider
When proposing a change to `Optim.jl`, there are a few things to consider. If you're in doubt feel free to reach out. A simple way to get in touch, is to join our [gitter channel](https://gitter.im/JuliaNLSolvers/Optim.jl).

Before submitting a pull request, please consider the following bullets:
* Did you remember to provide tests for your changes? If not, please do so, or ask for help.
* Did your change add new functionality? Remember to add a section in the documentation.
* Did you change existing code in a breaking way? Then remember to use Julia's deprecation tools to help users migrate to the new syntax.
* Add a note in the NEWS.md file, so we can keep track of changes between versions.

### Adding a solver
If you're contributing a new solver, you shouldn't need to touch any of the code in
`src/optimize.jl`. You should rather add a file named (`solver` is the name of the solver)
`solver.jl` in `src`, and make sure that you define an `Optimizer` subtype
`struct Solver <: Optimizer end` with appropriate fields, a default constructor with a keyword
for each field, a state type that holds all variables that are (re)used throughout
the iterative procedure, an `initial_state` that initializes such a state, and  an `update!` method
that does the actual work. Say you want to contribute a solver called
`Minim`, then your `src/minim.jl` file would look something like

```
struct Minim{IF, F<:Function, T} <: Optimizer
    alphaguess!::IF
    linesearch!::F
    minim_parameter::T
end

Minim(; alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang(), minim_parameter = 1.0) =
  Minim(linesearch, minim_parameter)

type MinimState{T,N,G}
  x::AbstractArray{T,N}
  x_previous::AbstractArray{T,N}
  f_x_previous::T
  s::AbstractArray{T,N}
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
