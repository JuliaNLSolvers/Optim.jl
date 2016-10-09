using Optim, ProgressMeter, Plots, DataFrames, BenchmarkTools

only_plots = true

default_names = ["Accelerated Gradient Descent",
                 "BFGS",
                 "L-BFGS",
                 "Momentum Gradient Descent",
                 "Conjugate Gradient",
                 "Gradient Descent",
                 "Nelder-Mead",
                 "Particle Swarm",
                 "Simulated Annealing",
                 "Newton",
                 "Newton (Trust Region)"]

default_solvers =[AcceleratedGradientDescent(),
                BFGS(),
                LBFGS(),
                ConjugateGradient(),
                GradientDescent(),
                MomentumGradientDescent(),
                NelderMead(),
                ParticleSwarm(),
                SimulatedAnnealing(),
                Newton(),
                NewtonTrustRegion()]

const cutest_problems = ["AKIVA", "ALLINITU", "ARGLINA", "ARGLINB",
    "ARGLINC", "ARWHEAD", "BARD", "BDQRTIC", "BEALE", "BIGGS6", "BOX", "BOX3",
    "BOXPOWER", "BRKMCC", "BROWNAL", "BROWNBS", "BROWNDEN", "BROYDN7D",
    "BRYBND", "CHAINWOO", "CHNROSNB", "CHNRSNBM", "CLIFF", "COSINE", "CRAGGLVY",
    "CUBE", "CURLY10", "CURLY20", "CURLY30", "DECONVU", "DENSCHNA", "DENSCHNB",
    "DENSCHNC", "DENSCHND", "DENSCHNE", "DENSCHNF", "DIXMAANA", "DIXMAANB",
    "DIXMAANC", "DIXMAAND", "DIXMAANE", "DIXMAANF", "DIXMAANG", "DIXMAANH",
    "DIXMAANI", "DIXMAANJ", "DIXMAANK", "DIXMAANL", "DIXMAANM", "DIXMAANN",
    "DIXMAANO", "DIXMAANP", "DIXON3DQ", "DJTL", "DQDRTIC", "DQRTIC", "EDENSCH",
    "EG2", "EIGENALS", "EIGENBLS", "EIGENCLS", "ENGVAL1", "ENGVAL2", "ERRINROS",
    "ERRINRSM", "EXPFIT", "EXTROSNB", "FLETBV3M", "FLETCBV2", "FLETCBV3",
    "FLETCHBV", "FLETCHCR", "FMINSRF2", "FMINSURF", "FREUROTH", "GENHUMPS",
    "GENROSE", "GROWTHLS", "GULF", "HAIRY", "HATFLDD", "HATFLDE", "HATFLDFL",
    "HEART6LS", "HEART8LS", "HELIX", "HIELOW", "HILBERTA", "HILBERTB",
    "HIMMELBB", "HIMMELBF", "HIMMELBG", "HIMMELBH", "HUMPS", "HYDC20LS",
    "INDEF", "INDEFM", "JENSMP", "JIMACK", "KOWOSB", "LIARWHD", "LOGHAIRY",
    "MANCINO", "MARATOSB", "MEXHAT", "MEYER3", "MODBEALE", "MOREBV", "MSQRTALS",
    "MSQRTBLS", "NCB20", "NCB20B", "NONCVXU2", "NONCVXUN", "NONDIA", "NONDQUAR",
    "NONMSQRT", "OSBORNEA", "OSBORNEB", "OSCIGRAD", "OSCIPATH", "PALMER1C",
    "PALMER1D", "PALMER2C", "PALMER3C", "PALMER4C", "PALMER5C", "PALMER6C",
    "PALMER7C", "PALMER8C", "PARKCH", "PENALTY1", "PENALTY2", "PENALTY3",
    "POWELLSG", "POWER", "QUARTC", "ROSENBR", "S308", "SBRYBND", "SCHMVETT",
    "SCOSINE", "SCURLY10", "SCURLY20", "SCURLY30", "SENSORS", "SINEVAL",
    "SINQUAD", "SISSER", "SNAIL", "SPARSINE", "SPARSQUR", "SPMSRTLS",
    "SROSENBR", "SSBRYBND", "SSCOSINE", "STRATEC", "TESTQUAD", "TOINTGOR",
    "TOINTGSS", "TOINTPSP", "TOINTQOR", "TQUARTIC", "TRIDIA", "VARDIM",
    "VAREIGVL", "VIBRBEAM", "WATSON", "WOODS", "YATP1LS", "YATP2LS", "YFITU",
    "ZANGWIL2"]

# Get the latest commit as a string due to Ismael Venegas Castelló (@Ismael-VC)
"""
Returns the shortened SHA of the latest commit of a package.
"""
function latest_commit(pkg::String)::String
    original_directory = pwd()
    cd(Pkg.dir(pkg))
    commit = readstring(`git rev-parse --short HEAD`) |> chomp
    cd(original_directory)
    return commit
end

"""
Returns, for each solver, a vector of counts of the number of problems in the DataFrame
that is below the threshold values given.
"""
function profiles(names, df, tau, measure)
    _profiles = []
    for name in default_names
        profile = zeros(tau)
        rows =  df[:Optimizer].==name
        for i = 1:length(tau)
            profile[i] = sum(df[rows, measure].<=tau[i])/length(unique(df[:Problem]))
        end
        push!(_profiles, profile)
    end
    _profiles
end

pkg_dir = Pkg.dir("Optim")
version_sha = latest_commit("Optim")
benchmark_dir = pkg_dir*"/benchmarks/"
version_dir = benchmark_dir*version_sha
try
    run(`mkdir $version_dir`)
catch

end
cd(version_dir)

!only_plots && include(benchmark_dir*"/CUTEst.jl")
!only_plots && include(benchmark_dir*"/Optim.jl")

str = version_dir*"/cutest_benchmark.csv"
df = readtable(str)
#=
str = Pkg.dir("Optim")*"/benchmarks/"*"$(version_sha)/optim_benchmark.csv"
df = readtable(str)
dff_agg = aggregate(df[[:Problem, :Minimum]], :Problem, minimum)
dff = join(df, dff_agg, on = :Problem)
df[:error] = (dff[:Minimum]-dff[:Minimum_minimum])=#
tau = 10.0.^(-32:10)

str = version_dir*"/cutest_benchmark.csv"
df = readtable(str)
f_profiles = profiles(default_names, df, tau, :f_error)
f_err = plot(tau, hcat(f_profiles...),
        label = hcat(default_names...),
        lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
        ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
        size =(800,400),
        ylims = (0,1),
        line = :steppre, xscale=:log, xlabel = "Error level", ylabel = "Proportion of problems",
        title = "Measure: f-f*")
savefig("f_err_cutest")

x_profiles = profiles(default_names, df, tau, :x_error)
x_err = plot(tau, hcat(x_profiles...),
        label = hcat(default_names...),
        lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
        ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
        size =(800,400),
        ylims = (0,1),
        line = :steppre, xscale=:log, xlabel = "Error level", ylabel = "Proportion of problems",
        title = "Measure: sup-norm of x-x*.")
savefig("x_err_cutest")

str = version_dir*"/optim_benchmark.csv"
df = readtable(str)
f_profiles = profiles(default_names, df, tau, :f_error)
f_err = plot(tau, hcat(f_profiles...),
        label = hcat(default_names...),
        lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
        ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
        size =(800,400),
        ylims = (0,1),
        line = :steppre, xscale=:log, xlabel = "Error level", ylabel = "Proportion of problems",
        title = "Measure: f-f*")
savefig("f_err_optim")

x_profiles = profiles(default_names, df, tau, :x_error)
x_err = plot(tau, hcat(x_profiles...),
        label = hcat(default_names...),
        lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
        ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
        size =(800,400),
        ylims = (0,1),
        line = :steppre, xscale=:log, xlabel = "Error level", ylabel = "Proportion of problems",
        title = "Measure: sup-norm of x-x*.")
savefig("x_err_optim")
