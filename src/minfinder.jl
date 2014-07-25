### MinFinder ###
# Syntax:
#    minima, fcount, searches, iterations = minfinder(func, l, u)
#    minima, fcount, searches, iterations = minfinder(func, l, u, opts)
# Inputs:
#   `func` must have syntax
#      val = func(g, x)
#   where g is storage for the gradient (or nothing, if the
#   	gradient is not desired)
#   l contain the lower boundaries of the search domain
#   u contain the upper boundaries of the search domain
# Outputs:
#	minima is a Vector that contains SearchPoint types
#   fcount is the number of function evaluations
#	searches is the number of local minimizations performed
#	iterations is the number of minfinder iterations before stopping rule hit
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
# aims to locate all the local minima of a multidimensional continuous and 
# differentiable function inside a bounded domain. [..] We compare the 
# performance of this new method to the performance of Multistart and 
# Topographical Multilevel Single Linkage Clustering on a set of benchmark 
# problems."
#
# Because the search domain is bounded, minfinder uses fminbox for local 
# searches and therefore the cgdescent function API.
#
# TODO: is the check that a new minima was already found correct (norm<tol)?
# TODO: the 2008 paper introduces non-gradient based checking rules. Thus a
#		derivative-free MinFinder could be implement that also uses derivative-
#		free local searches

# Create types for starting points and minima
type SearchPoint{T}
	x::Vector{T} # point
	g::Vector{T} # gradient, can be nothing
	f::T          # function value
end
SearchPoint{T}(x::Vector{T}, g::Vector{T}) = SearchPoint(x, g, nan(T))
SearchPoint{T}(x::Vector{T}) = SearchPoint(x, Array(T,0))
type Points{T}
	s::Vector{SearchPoint{T}}
end
Points(T) = Points{T}(Array(SearchPoint{T},0))
Base.push!{T}(P::Points{T}, S::SearchPoint{T}) = push!(P.s, S)
Base.deleteat!{T}(P::Points{T}, i::Integer) = deleteat!(P.s, i)
Base.length{T}(P::Points{T}) = length(P.s)
#Base.getindex{T}(P::Points{T}, i::Integer) = getindex(P.s, i)
#Base.setindex!{T}(P::Points{T}, S::SearchPoint{T}, i::Integer) = setindex!(P.s, S, i)

function minfinder{T}(func::Function, l::Array{T,1}, u::Array{T,1},ops::Options)

	# Set default options:
	#	stopping tolerances are set quite high by default (sqrt of usual tol); 
	# 	at the end the minima are polished off. Inspiration from S. Johnson:
	#	[http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#MLSL_.28Multi-Level_Single-Linkage.29]
	@defaults ops localtol=sqrt(eps(T)^(2/3)) polishtol=eps(T)^(2/3) ENRICH=1.1 NMAX=100 NINIT=20 EXHAUSTIVE=.5 show_iter=0 polish=true

	# Algortihm parameters as in paper:
	# NMAX = "predefined upper limit for the number of samples in each 
	# 		 generation. This  step prevents the algorithm from performing an 
	#		 insufficient exploration of the search space."
	# NINIT = initial number of samples 
	# EXHAUSTIVE =  "..in the range (0,1). For small values of p (p→0) the 
	# 		algorithm searches the area exhaustively, while for p→1, the 
	#		algorithm terminates earlier, but perhaps prematurely."

	# Other options:
	# localtol: tolerance level for local searches
	# polishtol: tolerance level for final polish of minima
	# polish: use final polish? (default=true)

	# Initiate
	length(l) == length(u) ||error("boundary vectors must have the same length")

	N = NINIT # number of starting point samples 
	TypicalDistance = 0. #Float64 because averaging
	MinDistance = Inf #for use in ValidPoint: min distance between found minima	
	stoplevel = 0. # 'a' in paper = EXHAUSTIVE * var_last

	minima = Points{T} # type with found minima
	polishminina = Points{T} # mimina after final polish
	iterminima = Points{T} #minima found during one iteration
	points = Points{T} #starting points for local minimizations

	gfunc = similar(l) # temporary jacobian for use inside `func`
	px = similar(l) # temporary point input
	pval = zero(T) # temporary function value
	pg = similar(l) # temporary point function gradient 

	fcount::Int = 0 #number of function evaluations
	iteration::Int =0 #number of minfinder iterations
	searches::Int=0 #number of local minimizations
	converges::Int=0 #number of converged searches

	fminops = @options tol=localtol # options for local minimizations
	polishops = @options tol=polishtol # options for final polish minimizations

	# Define stopping rule of paper. In short, create a series of binomial
	# events from 1 to N. The variance of this series goes slowly to zero. 
	# Compare this value with `stoplevel` at the latest iteration when a minima 
	# was found. 
	DoubleBox(n::Int) = var([rand(Binomial(i))/i for i=1:n]) 

	# Checking rules for each point before starting local searches
	function ValidPoint(p, pnts, mins)
		length(minima) == 0 && return true # no TypicalDistance without minima

		# condition 1: check against all other points, order matters
		for q in pnts
			norm(p.x - q.x, 2) < TypicalDistance &&
				(p.x - q.x)'*(p.g - q.g) > 0 && return false
		end

		# condition 2: check against found minima
		for z in minima
			norm(p.x - z.x, 2) < MinDistance && 
				(p.x - z.x)'*(p.g - z.g) > 0 &&	return false
		end
		return true
	end

	# Show progress
	if show_iter > 0 
		@printf "########### minfinder ########### \n"
		@printf "Iter   N    Searches   Function Calls   Minima \n"
	    @printf "----   ---  --------   --------------   ------ \n"
	end
    
	# main iteration loop
	while !(DoubleBox(N) < stoplevel)
		iteration += 1

		# Sampling and checking step
		empty!(points)
		for unused=1:N
			px = l + rand(length(l)).*(u-l)
			if length(minima) == 0 # no checks possible, gradient not required.
				push!(points, SearchPoint(px))
			else
				pval = func(pg, px)
				fcount += 1
				p = SearchPoint(p, pg, pval)
				ValidPoint(p, points, minima) && push!(points, p)
			end
		end

		# Enrichment for next iteration
		if length(points) < N/2 
			N = int(N * ENRICH)
		end

		empty!(iterminima)
		for p in points
			
			# if minima found during this iteration, check point against these.
			nextpoint = false
			if !isempty(iterminima)
				for z in iterminima
					if norm(p.x - z.x, 2) < MinDistance && 
							(p.x - z.x)'*(p.g - z.g) > 0
						nextpoint = true
					end
					nextpoint && continue # skip other minima checks
				end
				nextpoint && continue # skip local search for this point
			end

			# local minimization
			px, pval, fmincount, converged = fminbox(func, p.x, l, u, fminops)
			fcount+=fmincount
			searches += 1

			if converged
				converges +=1

				# Update typical search distance 
				TypicalDistance = (TypicalDistance*(searches - 1) + 
					norm(p.x - px, 2)) / searches
				
				# See if minima already found, if not, add to minimalists
				if !any([norm(px - m.x, 2) < localtol for m in minima])

					# Update stoplevel
					stoplevel = EXHAUSTIVE * DoubleBox(N) 

					# Update typical minima distance
					MinDistance = min(MinDistance, 
						[norm(px-m.x, 2) for m in minima]...)

					# Gradient not given as output fminbox, needs extra function
					# evaluation.
					pval = func(pg, px)
					fcount += 1				  
					push!(iterminima, SearchPoint(px, pg, pval))
				#Add also to global minima, to check next minima in iteration
					push!(minima, SearchPoint(px, pg, pval))
				end #if minima found
			end #if converged
		end #for points

		if show_iter > 0 
			@printf "%4d   %3d   %8d   %14d   %6d\n" iteration N searches fcount length(minima)
		end
	end #while

	# Polish off minima
	if polish
		for m in minima
			px, pval, fpolishcount, converged=fminbox(func, m.x, l, u,polishops)
			fcount += fpolishcount
			if !any([norm(px - h.x, 2) < polishtol for h in polishminina])
				pval = func(pg, px)
				fcount += 1
				push!(polishminina, SearchPoint(px, pg, pval))
			end
		end
		
		if show_iter > 0 
			@printf "Final polish retained %d minima out of %d \n" length(polishminima) length(minima)
		end

		return polishminima, fcount, searches, iteration
	else
		return minima, fcount, searches, iteration
	end
end

minfinder{T}(func::Function, x::Array{T}, l::Array{T}, u::Array{T}) = 
	minfinder(func, x, l, u, Options())
export minfinder



