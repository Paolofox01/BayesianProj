
using Pkg, Turing, StatsPlots
using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices
using Plots, CSV, HDF5
using ForwardDiff

include("utilities_turing.jl")
include("simulated_data.jl")

# lettura dei dati
df = CSV.read("FinalStations.csv", DataFrame; delim=',')

# Parametri 
K = 2       #numero di fonti
C = 6       # inquinanti
N = 32      #numero di siti
T = 30      #365 # istanti temporali
n_cov = 4   # covariate x

# Set a fixed seed
seed = 1997

    
# generating simulated data up to g_ik
sites, dat, theta_true, dat_trials = simulate_data(df, seed, K, N, T)
sizeof(dat)
names(dat)
size(dat[:f])
size(dat[:g])
size(dat[:gamma])
# generating y_ic
sample_y = generate_y(dat[:g], seed) # generazione y dato le g generate

# matrix of distances
dist = euclid_dist(sites[:, 1], sites[:, 2], N)
maximum(dist)
#issymmetric(dist)

# setting time instances
days = range(1, stop=T, length=T)

# design matrix
X = sites[:,3:6]


# TODO: plot degli y per una location. 
#       Scegliamo una location i* e facciamo multiplot con 6 panels: un panel per time-series di ogni inquinante c=1,..,6


# modello Turing
@model function gp_latent_model(y, K, dist, X, days, ::Type{TV}=Float64) where {TV}

    N, T, C = size(y)  # Individui x Tempo x Inquinanti
    n_cov = size(X,2)
    # PARTE I: temporale
    println("\n new iter \n")

    rho_time ~ Gamma(1, 0.25)
    Sigma_f = sq_exp_kernel(days, rho_time, nugget = 1e-9)
    Sigma_f_inv = inv(Sigma_f)
    #println("Sigma_f posdef: ", isposdef(Sigma_f))

    f = Matrix{Float64}(undef, T, K)  # Preallocazione per f
    for k in 1:K
        f[:, k] ~ MvNormal(zeros(T), Sigma_f)  # GP temporale
    end

    # PARTE 2: spaziale

    
    rho_space = Vector{Any}(undef, K)  # Preallocazione per K
    K_space = Array{Any}(undef, N, N, K)
    for k in 1:K
        #rho_space[k] ~ Gamma(1, 0.2)
        rho_space[k] = 0.0024
        K_space[:,:,k] = get_Sigma_gamma(dist, rho_space[k])
    end
    
    K_space = Float64.(K_space)
    #gamma_ln = Matrix{Float64}(undef, N, K)
    gamma = Matrix{Float64}(undef, N, K)
    beta = Matrix{Float64}(undef, n_cov, K)  
    mu_gamma = Matrix{Any}(undef, N, K)
    for k in 1:K
        beta[:,k] ~ MvNormal(zeros(n_cov), I(n_cov))
        mu_gamma[:,k] = X[:, :]*beta[:,k]
        # in gamma_ln tenere "Vector{Float64}" altrimenti ci sarà errore
        gamma[:, k] ~ MvNormal(eltype(beta[1,1]).(mu_gamma[:,k]), K_space[:,:,k])  
    end
    #gamma = exp.(gamma_ln)  # ora avremo e^gamma

    
    # PARTE III: costruzione delle g's

    tau ~ MvNormal(zeros(N), tau_covariance(N, 1)) # 3 preso dal code iniziale, nel main_new

    g = Array{Float64, 3}(undef, N, T, K)
    mu_g = Array{Any}(undef, T)
    var_g = Matrix{Any}(undef,T,T)
    for k in 1:K
        println("k = ", k)
        for i in 1:N
            mu_g = get_mu_g(i, days, f[:, k], Dict(:rho => rho_time, :tau => tau, :gamma => gamma[:,k]), Sigma_f_inv)

            var_g = get_Sigma_g_i(i, days, Dict(:rho => rho_time, :tau => tau, :gamma => gamma[:,k]), Sigma_f, Sigma_f_inv)
            #println("i = ", i, "; var_g posdef: ", isposdef(var_g))
            if !isposdef(var_g)
                println("var_g symm: ", issymmetric(var_g))
                println(var_g)
                println(Dict(:rho => rho_time, :tau => tau, :gamma => gamma[:,k]))
            end
            #g[i, :, k] ~ MvNormal( mu_g , I(T)) # mettere var_fin quando risolviamo Hermitian Matrix error
            mu_g = eltype(beta[1,1]).(mu_g)
            var_g = eltype(beta[1,1]).(var_g)
            g[i, :, k] ~ MvNormal( mu_g , var_g)
        end
    end


    # PARTE IV: costruzione y's

    mu_y = Array{Any}(undef, N, T, C)
    sigma_y = Array{Float64}(undef, C)
    h = Matrix{Any}(undef,K,C)
    for k in 1:K
        h[k,:] ~ Dirichlet(ones(C)) 
    end

    for c in 1:C
        sigma_y[c] ~ InverseGamma(3, 2)

        for t in 1:T
            for i in 1:N

                mu_y[i,t,c] = sum(h[:,c] .* g[i,t,:])
                mu_y[i,t,c] = eltype(beta[1,1]).(mu_y[i,t,c])        
                y[i,t,c] ~ Normal(mu_y[i,t,c], sigma_y[c])
                
            end
        end
    end
end

# Start sampling.
num_iterations = 10
Random.seed!(seed)

#chainTrue = sample(gp_latent_model(sample_y, K, dist, X[:,1:3], days), Prior(), num_iterations) # ok 
chainTrue = sample(gp_latent_model(sample_y, K, dist, X[:,1:3], days), HMC(0.1,10), num_iterations)
# ora funziona ma fa loop infinito: fare check!

size(chainTrue)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
println(names(chainTrue)[1])
histogram(chainTrue[:"rho_time"])






## TEST per vedere se "var_fin" funziona ---> FUNZIONA

N, T, C = size(sample_y)  # Individui x Tempo x Inquinanti


# PARTE I: temporale

rho_time = rand(Gamma(5, 1))
Sigma_f = sq_exp_kernel(days, rho_time)
Sigma_f_inv = inv(Sigma_f)

f = Matrix{Float64}(undef, T, K)  # Preallocazione per f
for k in 1:K
    f[:, k] = rand(MvNormal(zeros(T),Sigma_f))  # GP temporale
end


# PARTE 2: spaziale

dist = euclid_dist(sites[:, 1], sites[:, 2], N)
K_space = Array{Float64}(undef, N,N, K)
rho_space = Array{Float64}(undef, K)  # Preallocazione per K
for k in 1:K
    rho_space[k] = rand(Gamma(3, 1000))
    K_space[:,:,k] = exp.(- (1 ./ rho_space[k]) .* dist)
end

gamma_ln = Matrix{Float64}(undef, N, K)
beta = Matrix{Float64}(undef, n_cov, K)  
for k in 1:K
    beta[:,k] = rand(MvNormal(zeros(n_cov), I(n_cov)))

    # in gamma_ln tenere "Vector{Float64}" altrimenti ci sarà errore
    gamma_ln[:, k] = rand(MvNormal(Vector{Float64}(X[:, :]*beta[:,k]), K_space[:,:,k])  )
end
gamma = exp.(gamma_ln)  # ora avremo e^gamma


# PARTE III: costruzione delle g's

tau = rand(MvNormal(zeros(N), tau_covariance(N,3))) # 3 preso dal code iniziale, nel main_new

g = Array{Float64, 3}(undef, N, T, K)
mu_g = Array{Float64}(undef, T)
var_g = Matrix{Float32}(undef,T,T)

for k in 1:K
    for i in 1:N

        K_i = get_K_i(days, Dict(:rho => rho_time, :tau => tau[i], :gamma => gamma[i, k]))

        mu_g = Vector{Float64}(K_i* Sigma_f_inv * f[:, k]) # Tx1
        var_g = (gamma[i,k]^2)*Sigma_f - K_i* Sigma_f_inv *K_i' + 10^(-1)*I(T) # TxT
        
        var_fin = 0.5*(var_g+var_g') # per simmetrizzare, per ora non funziona
        
        g[i, :, k] = rand(MvNormal( mu_g , var_fin)) # mettere var_fin quando risolviamo Hermitian Matrix error
    
    end
end
