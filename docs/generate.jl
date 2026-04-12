# generate examples
import Literate: Literate, CommonMarkFlavor, DocumenterFlavor

# TODO: Remove items from `SKIPFILE` as soon as they run on the latest
# stable `Optim` (or other dependency)
#ONLYSTATIC = ["ipnewton_basics.jl",]
ONLYSTATIC = []

EXAMPLEDIR = joinpath(@__DIR__, "src", "examples")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")
for example in filter!(endswith(".jl"), readdir(EXAMPLEDIR))
    input = abspath(joinpath(EXAMPLEDIR, example))
    script = Literate.script(input, GENERATEDDIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    execute = !(example in ONLYSTATIC)
    Literate.markdown(
        input,
        GENERATEDDIR,
        postprocess = mdpost,
        flavor = execute ? DocumenterFlavor() : CommonMarkFlavor(),
    )
    Literate.notebook(input, GENERATEDDIR; execute)
end
