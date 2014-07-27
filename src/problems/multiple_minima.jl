module MultipleMinimaProblems

# Test problems with multiple minima from paper and from:
# http://www2.compute.dtu.dk/~kajm/Test_ex_forms/test_ex.html

immutable OptimizationProblem
    name::ASCIIString
    f::Function
    l::Vector
    u::Vector
    min::Vector{Vector} # all local minima
    # glob_x::Vector{Float64} # set of global minimum points 
    glob_f::FloatingPoint # global minimum function value
end

examples = Dict{ASCIIString, OptimizationProblem}()

## Rosenbrock ##
examples["Rosenbrock"] = OptimizationProblem(
	"Rosenbrock",
    function rosenbrock(g, x::Vector)
    	d1 = 1.0 - x[1]
    	d2 = x[2] - x[1]^2
	    if !(g === nothing)
	    	g[1] = -2.0*d1 - 400.0*d2*x[1]
	    	g[2] = 200.0*d2
	    end
	    val = d1^2 + 100.0 * d2^2
	    return val
  	end,
  	[-2., -2],
  	[2.,2],
  	{[1.,1.]},
  	0.)
	
## Camel ##
examples["Camel"] = OptimizationProblem(
	"Camel",
    function camel(g, x::Vector)
        if !(g === nothing)            g[1] = 8x[1] - 8.4x[1]^3 + 2x[1]^5 + x[2]
            g[2] = x[1] - 8x[2] + 16x[2]^3
        end
        return 4x[1]^2 - 2.1x[1]^4 + 1/3*x[1]^6 + x[1]*x[2] - 4x[2]^2 + 4x[2]^4
    end,
    [-5., -5],
    [5., 5],
	{[-0.0898416,0.712656],[0.089839,-0.712656], [-1.6071,-0.568651], 
	   [1.70361,-0.796084], [1.6071,0.568652], [-1.70361,0.796084]},
	-1.03163)

## Rastrigin ##
examples["Rastrigin"] = OptimizationProblem(
	"Rastrigin",
    function rastrigin(g, x)
        if !(g === nothing)
            g[1] = 2x[1] +18sin(18x[1])
            g[2] = 2x[2] +18sin(18x[2])
        end
        return x[1]^2 + x[2]^2 - cos(18x[1]) - cos(18x[2])
    end,
    [-1, -1],
    [1, 1],
	{[0.69384,-0.34693],[-0.34693,0.34692],[0.69384,0.99999],[-0.69385,-0.34693],
        [0.34692,0.34692],[-0.69385,-1.0],[0.0,-0.69385],[-0.69385,0.99999],
        [0.0,-1.0e-5],[-0.69385,0.34692],[0.69384,-0.69385],[-1.0e-5,-0.34693],
        [0.0,0.69384],[0.34692,0.0],[0.34692,-0.69385],[0.34692,0.99999],
        [-0.34693,0.69384],[-0.34693,0.0],[-1.0e-5,-1.0],[0.99999,-0.69385],
        [0.34692,-1.0],[0.69384,0.34692],[0.0,0.99999],[-0.34693,-0.69385],
        [0.99999,0.34692],[0.69384,-1.0],[0.99999,-1.0],[-0.69385,0.0],
        [-0.69385,0.69384],[0.34692,-0.34693],[0.69384,0.0],[-0.34693,-0.34693],
        [-0.69385,-0.69385],[-1.0e-5,0.34692],[-1.0,0.0],[0.34692,0.69384],
        [-1.0,-1.0],[0.99999,0.0],[0.69384,0.69384],[-1.0,-0.34693],
        [-1.0,-0.69385],[0.99999,-0.34693],[-0.34693,0.99999],[-0.34693,-1.0],
        [0.99999,0.69384],[-1.0,0.69384],[-1.0,0.34692],[-1.0,0.99999],
        [0.99999,0.99999]},
    -1.03163)


end # module