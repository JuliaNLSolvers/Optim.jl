### MinFinder ###
# Syntax:
#    minimavec, minimavalvec, fcount = minfinder(func, l, u)
#    minimavec, minimavalvec, fcount = minfinder(func, l, u, opts)
# Inputs:
#   func must have syntax
#      val = func(g, x)
#   where g is storage for the gradient (or nothing, if the
#   gradient is not desired)
#   l contain the lower boundaries of the search domain
#   u contain the upper boundaries of the search domain
# Outputs:
#	minimavec is the vector with minima locations
#	minimavalvec contains the minima values
#	minimagradmatrix contains the gradients at the minima 
#   fcount is the number of function evaluations
# 
# Based on the papers:
# Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder: Locating all the local minima
# of a function, Computer Physics Communications, Volume 174, January 2006, 
# Pages 166-179. http://dx.doi.org/10.1016/j.cpc.2005.10.001
#
# Ioannis G. Tsoulos, Isaac E. Lagaris, MinFinder v2.0: An improved version of 
# MinFinder, Computer Physics Communications, Volume 179, Issue 8, 
# 15 October 2008, Pages 614-615, ISSN 0010-4655
# http://dx.doi.org/10.1016/j.cpc.2008.04.016.
#
# From the abstract: "A new stochastic clustering algorithm is introduced that 
# aims # to locate all the local minima of a multidimensional continuous and 
# differentiable function inside a bounded domain. [..] We compare the 
# performance of this new method to the performance of Multistart and 
# Topographical Multilevel Single Linkage Clustering on a set of benchmark 
# problems."
#
# Because the search domain is bounded, minfinder uses fminbox for local 
# searches and the cgdescent function API.
#
# TODO: the 2008 paper introduces non-gradient based checking rules and a
#		derivative-free MinFinder could be implement that also uses derivative-
#		free local searches



function minfinder{T}(func::Function, l::Array{T,1}, u::Array{T,1}, ops::Options)

	# Set default options:
	#	stopping tolerances are set quite high by default (sqrt of usual tol); 
	# 	at the end the minima are polished off. Inspiration from:
	#	[http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#MLSL_.28Multi-Level_Single-Linkage.29]
	@defaults ops tol=sqrt(eps(T)^(2/3)) ENRICH=1.1 NMAX=100 NINIT=20 EXHAUSTIVE=.5

	# Algortihm parameters as in paper:
	# NMAX = "predefined upper limit for the number of samples in each 
	# 		 generation. This  step prevents the algorithm from performing an 
	#		 insufficient exploration of the search space."
	# NINIT = initial number of samples 
	# EXHAUSTIVE =  "..in the range (0,1). For small values of p (p→0) the 
	# 		algorithm searches the area exhaustively, while for p→1, the 
	#		algorithm terminates earlier, but perhaps prematurely."

	# Create Type for starting points and minima
	type SearchPoint
		x::Array{T,1}
		g::Array{T,1}
		f::T
	end
	SearchPoint(x::Array{T,1}, g::Array{T,1}) = SearchPoint(x, g, nothing)
	SearchPoint(x::Array{T,1}) = SearchPoint(x, nothing)

	# Initiate
	length(l) == length(u) ||error("boundary vectors must have the same length")
	dim	= length(l) # dimension of the function input
	N = NINIT # number of starting point samples 
	TypicalDistance = 0. #Float64 because averaging
	MinDistance = 0. #for use in ValidPoint: min distance between found minima	
	stoplevel = 0. # 'a' in paper = EXHAUSTIVE * var_last
	minima = Array(Vector{SearchPoint},0) # found minima
	iterminima = Array(Vector{T}, 0) #minima found during one iteration
	points = Array(Vector{T}, 0) #starting points for local minimazations
	gfunc = zeros(T, dim) # temporary jacobian for use inside `func`
	px = similar(gfunc) # temporary point input
	pval = zero(T) # temporary point function value
	pg = simimlar(p) # temporary point function gradient 
	fcount::Int = 0 #number of function evaluations
	iteration::Int =0 #number of minfinder iterations
	# xmins = Array(Vector{T}, 0) #contains found minima
	# xgrads= Array(Vector{T}, 0) #gradients of found minima
	# pstart = Array(Vector{T}, 0)#points to start local optimization
	# pgrads = Array(Vector{T}, 0)#gradients of stating points

	# Define stopping rule of paper. In short, create a series of binomial
	# events up to N. The variance of this series goes slowly to zero. Compare
	# this value with `stoplevel` of the latest iteration when a minima was 
	# found. 
	DoubleBoxStop(n::Int, stop) = var([rand(Binomial(i))/i for i=1:n]) < 
		stoplevel

	# Checking rules for all points before starting a local search
	function CheckPoints(points)
		length(minima) == 0 && return

		# condition 1: check against other points
		for i=1:length(pstart) - 1 
			for j=i:length(pstart)
				if norm(pstart[i] - pstart[j], 2) < TypicalDistance &&
						(pstart[i] - pstart[j])'*(pgrads[i] - pgrads[j]) > 0
					
					deleteat!(pstart, i)
					deleteat!(pgrads, i)
				end
			end
		end
		# condition 2: check against found minima
		for i=1:length(pstart) 
			for j=1:length(xmins)
				if norm(pstart[i]-xmins[j], 2) < MinDistance && 
						(pstart[i] - xmins[j])'*(pgrads[i] - grads[j]) > 0
 					
 					deleteat!(pstart, i)
					deleteat!(pgrads, i)
			end
		end
	end


	# main iteration loop
	while !DoubleBoxStop(N, stoplevel)
		iteration += 1

		# Sampling and checking step
		empty!(points)
		# empty!(pstart)
		# empty!(pgrads)
		for unused=1:N
			p = l + rand(dim).*(u-l)
			if length(minima) == 0
				push!(points, SearchPoint(p))
			else
				pval = func(pg,p)
				fcount += 1
				push!(points, SearchPoint(p, pg, pval))
			end
			# push!(pstart, p)
			# push!(pgrads, pg)
		end
		CheckPoints(points)

		# Enrichment for next iteration
		if length(pstart) < N/2 
			N = int(N * ENRICH)
		end

		empty!(itermins)
		for p_i=1:length(pstart)







end
minfinder{T}(func::Function, x::Array{T}, l::Array{T}, u::Array{T}) = 
	minfinder(func, x, l, u, Options())
export minfinder



