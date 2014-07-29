### MinFinder ###
# Syntax:
#    minima, fcount, searches, iterations = minfinder(func, l, u)
# Inputs:
#   `func` must have syntax
#      val = func(g, x)
#   where g is storage for the gradient (or nothing, if the
#       gradient is not desired)
#   l contain the lower boundaries of the search domain
#   u contain the upper boundaries of the search domain
# Outputs:
#    minima is a Vector that contains SearchPoint types
#    fcount is the number of function evaluations
#    searches is the number of local minimizations performed
#    iterations is the number of minfinder iterations before stopping rule hit
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
#        derivative-free MinFinder could be implement that also uses derivative-
#        free local searches
# TODO: add parallel computing

# Create types for the starting points and the minima

type SearchPoint{T <: Number} 
    x::Vector{T} # point
    g::Vector{T} # gradient, can be nothing
    f::T         # function value
end
SearchPoint{T}(x::Vector{T}, g::Vector{T}) = SearchPoint(x, g, nan(T))
SearchPoint{T}(x::Vector{T}) = SearchPoint(x, Array(T,0))

function minfinder{T <: FloatingPoint}(func::Function, l::Array{T,1}, u::Array{T,1};
    ENRICH = 1.1,
    NMAX::Int = 250,
    NINIT::Int = 20,
    EXHAUSTIVE = .5,
    max_iter::Int = 1_000,
    show_trace::Bool = false,
    polish::Bool = true, 
    local_tol = (polish ? sqrt(eps(T)^(2/3)) : eps(T)^(2/3)),
    polish_tol = eps(T)^(2/3),
    distmin = sqrt(local_tol),
    distpolish = sqrt(polish_tol))

    # When using polish, stopping tolerances are set quite high by default 
    # (sqrt of usual tol). At the end the minima are polished off. Inspiration 
    # from S. Johnson: [http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#MLSL_.28Multi-Level_Single-Linkage.29]

    # Algortihm parameters as in paper:
    # NMAX = "predefined upper limit for the number of samples in each 
    #        generation. This  step prevents the algorithm from performing an 
    #        insufficient exploration of the search space."
    # NINIT = initial number of samples 
    # EXHAUSTIVE =  "..in the range (0,1). For small values of p (p→0) the 
    #         algorithm searches the area exhaustively, while for p→1, the 
    #         algorithm terminates earlier, but perhaps prematurely."

    # Other options:
    # polish: Perform final optimization on each found minima?
    # local_tol: tolerance level for local searches
    # polish_tol: tolerance level for final polish of minima    
    # distmin: discard minima is closer than distmin to found minima
    # distpolish: same for final minima polish
    # max_iter: maximum number of iterations (each with N points sampled)
    # show_iter: show progress if >0

    # Initiate
    length(l) == length(u) || error("boundary vectors must have the same length")

    N = NINIT # number of starting point samples 
    typical_distance = zero(T) # typical distance between start and its minima
    min_distance = inf(T) #for use in ValidPoint: min distance between minima    
    stoplevel = 0. # 'a' in paper = EXHAUSTIVE * var_last

    minima = Array(SearchPoint{T}, 0) # type with found minima
    polishminina = Array(SearchPoint{T}, 0) # mimina after final polish
    iterminima = Array(SearchPoint{T}, 0) #minima found during one iteration
    points = Array(SearchPoint{T}, 0) #starting points for local minimizations

    gfunc = similar(l) # temporary jacobian for use inside `func`
    fx = similar(l) # temporary point input
    fval = zero(T) # temporary function value
    fg = similar(l) # temporary point function gradient 
    p = SearchPoint(fx, fg, fval) #temporary SearchPoint

    fcount::Int = 0 #number of function evaluations
    iteration::Int = 0 #number of minfinder iterations
    searches::Int = 0 #number of local minimizations
    converges::Int = 0 #number of converged searches

    fminops = @options tol=local_tol # options for local minimizations
    polishops = @options tol=polish_tol # options for final polish minimizations

    # Define stopping rule of the paper. In short, create a series of binomial
    # events from 1 to N. The variance of this series goes slowly to zero. 
    # Compare this value with `stoplevel` at the latest iteration when a minima 
    # was found. 
    #doublebox(n::Int) = var([StatsBase.rand_binom(i, .5)/i for i=1:n])
    # StatsBase.rand_binom does not work with julia v0.2.1, so sum bernoulli
    doublebox(n::Int) = var([sum(int(rand(i)))/i for i=1:n])

    dim = length(l) #precalc dimension of problem
    function checkrule{T}(a::SearchPoint{T}, b::SearchPoint{T}, dist)
        #L2dist(a.x, b.x) < dist && dot(a.x - b.x, a.g - b.g) > 0
        ax = a.x
        bx = b.x
        ag = a.g
        bg = b.g
        s = zero(T)
        t = zero(T)
        for i = 1:dim
            dx = ax[i] - bx[i]
            s += dx * dx
            t += dx * (ag[i] - bg[i])
        end
        return sqrt(s) < dist && t > 0
    end

    # Show progress
    if show_trace
        @printf "########### minfinder ########### \n"
        @printf "Iter   N    Searches   Function Calls   Minima \n"
        @printf "----   ---  --------   --------------   ------ \n"
    end
    
    # main iteration loop
    while (doublebox(N) > stoplevel) & (iteration < max_iter)
        iteration += 1

        # Sampling and checking step
        points = Array(SearchPoint{T}, 0) #empty points
        for unused=1:N
            fx = l + rand(dim) .* (u - l)
            fval::T = func(fg, fx)
            fcount += 1
            p = SearchPoint(fx, copy(fg), fval)

            # check on each point before accepting as starting point
            validpoint = true
            if !isempty(minima) # no typical_distance without minima        
                # condition 1: check against all other points in `pnts`
                for q in points
                    if checkrule(p, q, typical_distance); validpoint=false; end
                end
                # condition 2: check against found minima in `mins`
                for z in minima
                    if checkrule(p, z, min_distance);validpoint=false; end
                end
            end 
            validpoint && push!(points, p)
        end

        # Enrichment for next iteration
        if length(points) < N/2 
            N = min(int(N * ENRICH), NMAX)
        end

        iterminima = Array(SearchPoint{T}, 0) #empty iterminima
        for p in points

            # If minima found during this iteration, check point against these.
            nextpoint = false #TODO is there a way to break out of outer for?
            if !isempty(iterminima)
                for z in iterminima
                    if checkrule(p, z, min_distance); nextpoint = true; end
                    nextpoint && continue # skip other minima checks
                end
                nextpoint && continue # skip local search for this point
            end

            # local minimization
            fx, fvals, fmincount, converged = fminbox(func, p.x, l, u, fminops)
            fcount += fmincount
            searches += 1
            fval = minimum([minimum(fvals[i]) for i in 1:length(fvals)])

            if converged
                converges +=1
            
                # Update typical search distance 
                typical_distance = (typical_distance*(searches - 1) + 
                    norm(p.x - fx,2)) / searches
                #@printf "new typ distance: %s \n" typical_distance
                
                # Check if minima already found, if not, add to minimalists
                minfound = false
                for m in minima
                    if norm(fx - m.x,2) < distmin
                        minfound = true
                        continue
                    end
                end
                if !minfound
            
                    # Update stoplevel
                    stoplevel = EXHAUSTIVE * doublebox(N) 
            
                    # Update typical minima distance
                    if isempty(minima); min_distance = norm(fx - p.x,2); end
                    for m in minima
                       min_distance = min(min_distance, norm(fx - m.x,2))
                    end

                    # Gradient not given as output fminbox, needs extra function
                    # evaluation.
                    fval = func(fg, fx)
                    fcount += 1                  
                    push!(iterminima, SearchPoint(fx, copy(fg), fval))
                    #Add also to global minima, to check next minima in iteration
                    push!(minima, SearchPoint(fx, copy(fg), fval))
            
                end #if minima found
            end #if converged
        end #for points

        if show_trace 
            @printf "%4d   %3d   %8d   %14d   %6d\n" iteration N searches fcount length(minima)
        end
    end #while

    # Polish off minima
    if polish
        for m in minima
            # run final optization from each found minima
            fx, fvals, fpolishcount, converged = fminbox(func, m.x, l, u, polishops)
            fcount += fpolishcount

            # Check if not converges to another final optimization minima
            minfound = false
            for h in polishminina
                if norm(fx - h.x,2) < distpolish
                    minfound = true
                    continue
                end
            end
            if !minfound
                fval = func(fg, fx)
                fcount += 1
                push!(polishminina, SearchPoint(fx, fg, fval))
            end
        end
        
        if show_trace 
            @printf "Final polish retained %d minima out of %d \n" length(polishminina) length(minima)
        end

        return polishminina, fcount, searches, iteration
    else
        return minima, fcount, searches, iteration
    end #if polish

end #function

minfinder{T,S}(func::Function, l::Array{T,1}, u::Array{S,1};kwargs...) = 
    minfinder(func, [convert(Float64, i) for i in l], 
                    [convert(Float64, i) for i in u];kwargs...)
