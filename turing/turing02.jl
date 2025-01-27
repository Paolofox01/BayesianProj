
using Pkg, Turing, StatsPlots
using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices
using Plots, CSV, HDF5


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

    
# generating simulated data
sites, dat, theta_true, dat_trials = simulate_data(df, seed, K, N, T)
sizeof(dat)
names(dat)
size(dat[:g])

# MF: quiiiii

sample_y = generate_y(samples[:g], seed) # generazione y dato le g generate

# TODO: plot degli y per una location. 
#       Scegliamo una location i* e facciamo multiplot con 6 panels: un panel per time-series di ogni inquinante c=1,..,6

# modello Turing
@model function gp_latent_model(y, K, sites, car, days)

    N, T, C = size(y)  # Individui x Tempo x Inquinanti


    # PARTE I: temporale
    println("\n new iter \n")
    rho_time ~ Gamma(5, 1)
    K_f = sq_exp_kernel(days, rho_time, nugget = 1e-9)
    K_f_inv = inv(K_f)
    println("K_f posdef: ", isposdef(K_f))

    f = Matrix{Float64}(undef, T, K)  # Preallocazione per f
    for k in 1:K
        f[:, k] ~ MvNormal(zeros(T),K_f)  # GP temporale
    end


    # PARTE 2: spaziale

    dist = euclid_dist(sites[:, 1], sites[:, 2], N)
    K_space = Array{Float64}(undef, N,N, K)
    rho_space = Array{Float64}(undef, K)  # Preallocazione per K
    for k in 1:K
        rho_space[k] ~ Gamma(3, 1000)
        K_space[:,:,k] = exp.(- (1 ./ rho_space[k]) .* dist)
    end

    gamma_ln = Matrix{Float64}(undef, N, K)
    beta = Matrix{Float64}(undef, n_cov, K)  
    for k in 1:K
        beta[:,k] ~ MvNormal(zeros(n_cov), I(n_cov))

        # in gamma_ln tenere "Vector{Float64}" altrimenti ci sarà errore
        gamma_ln[:, k] ~ MvNormal(Vector{Float64}(car[:, :]*beta[:,k]), K_space[:,:,k])  
    end
    gamma = exp.(gamma_ln)  # ora avremo e^gamma

    
    # PARTE III: costruzione delle g's

    tau ~ MvNormal(zeros(N), tau_covariance(N,3)) # 3 preso dal code iniziale, nel main_new

    g = Array{Float64, 3}(undef, N, T, K)
    mu_g = Array{Float64}(undef, T)
    var_g = Matrix{Float32}(undef,T,T)

    for k in 1:K
        println("k = ", k)
        for i in 1:N

            K_i = get_K_i(days, Dict(:rho => rho_time, :tau => tau[i], :gamma => gamma[i, k]))

            mu_g = Vector{Float64}(K_i* K_f_inv * f[:, k]) # Tx1
            var_g = (gamma[i,k]^2).*K_f - K_i* K_f_inv *K_i' + 10^(-9)*I(T) # TxT
            var_fin = (var_g+var_g')/2 # per simmetrizzare, per ora non funziona
            println("i = ", i, "; var_fin posdef: ", isposdef(var_fin))
            if !isposdef(var_fin)
                println(var_fin)
            end
            #g[i, :, k] ~ MvNormal( mu_g , I(T)) # mettere var_fin quando risolviamo Hermitian Matrix error
            g[i, :, k] ~ MvNormal( mu_g , var_fin)
        end
    end


    # PARTE IV: costruzione y's

    mu = zeros(N, T, C)
    sigmay = zeros(C)
    h = Matrix{Float64}(undef,K,C)
    for k in 1:K
        h[k,:] ~ Dirichlet(ones(C)) 
    end

    for c in 1:C
        sigmay[c] ~ InverseGamma(3, 2)

        for t in 1:T
            for i in 1:N

                mu[i,t,c] = sum(h[:,c] .* g[i,t,:])
                        
                y[i,t,c] ~ Normal(mu[i,t,c], sigmay[c])
                
            end
        end
    end
end

# Start sampling.
iterations = 10
Random.seed!(seed)

chainTrue = sample(gp_latent_model(sample_y, K, sites, car, days), Prior(), iterations)
size(chainTrue)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
println(names(chainTrue)[1])
histogram(chainTrue[:"rho_time"])






## TEST per vedere se "var_fin" funziona ---> FUNZIONA

N, T, C = size(sample_y)  # Individui x Tempo x Inquinanti


# PARTE I: temporale

rho_time = rand(Gamma(5, 1))
K_f = sq_exp_kernel(days, rho_time)
K_f_inv = inv(K_f)

f = Matrix{Float64}(undef, T, K)  # Preallocazione per f
for k in 1:K
    f[:, k] = rand(MvNormal(zeros(T),K_f))  # GP temporale
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
    gamma_ln[:, k] = rand(MvNormal(Vector{Float64}(car[:, :]*beta[:,k]), K_space[:,:,k])  )
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

        mu_g = Vector{Float64}(K_i* K_f_inv * f[:, k]) # Tx1
        var_g = (gamma[i,k]^2)*K_f - K_i* K_f_inv *K_i' + 10^(-1)*I(T) # TxT
        
        var_fin = 0.5*(var_g+var_g') # per simmetrizzare, per ora non funziona
        
        g[i, :, k] = rand(MvNormal( mu_g , var_fin)) # mettere var_fin quando risolviamo Hermitian Matrix error
    
    end
end
