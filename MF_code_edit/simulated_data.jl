
using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities.jl")
include("fit_MCMC.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")


function simulate_data(df, seed, K, n, n_time)
    # Impostazione del seed
    Random.seed!(seed)

    # Parametri
    # K = 2 #numero di fonti
    # n = 32 #numero di siti
    # n_time = 30 #365

    intercept = ones(n)
    x1 = zeros(n)
    x2 = zeros(n)

    for cont in 1:n 
        if df[cont, 8] == "SUBURBAN"
            x1[cont] = 1
        end
        
        if df[cont, 8] == "URBAN AND CENTER CITY"
            x2[cont] = 1
        end

    end

    # TODO: standardizzare altitudine
    sites = Matrix(DataFrame(Latitude = df[:, 1], Longitude = df[:, 2], intercept = intercept, x1 = x1, x2=x2, Elevation = df[:, 3]))
    
    maximum(euclid_dist(sites[: , 1], sites[: , 2], n))

    # Parametri del modello
    theta = Dict{Int64, Dict{Any,Any}}()
    theta[1] = Dict(
        :rho => 0.1,
        :phi => 1/300.0,
        :gamma => zeros(32),
        :beta => [-0.5, 0.5, 1., 0.1],
        :tau => rand(Normal(0, 1), 32) 
    )

    theta[2] = Dict(
        :rho => 0.2,
        :phi => 1/400.0,
        :gamma => zeros(32),
        :beta => [0.3, -0.4, -0.7, -0.02],
        :tau => rand(Normal(0, 1), 32)
    )


    # Funzione per generare i dati (devi definire `generate_data` in Julia)
    dat = generate_data(sites, n, K, n_time, theta)

    # salvo i dati simulati per ogni k=1,..,K
    theta_true = copy(theta)
    df_new = Dict{Int64, DataFrame}()
    dat_trials = Dict{Int64, DataFrame}()
    for k in 1:K
        theta_true[k][:gamma] = dat[:gamma][:,k]
        df_new[k] = DataFrame(dat[:g][:, k, :], :auto)
        # "Melt" della matrice (equivalente a reshape2::melt in R)
        dat_trials[k] = stack(df_new[k], variable_name = "time", value_name = "value")
        # Aggiungiamo una colonna per il trial
        dat_trials[k].trial = repeat(1:n,n_time)
    end

   

    
    ##########################################################################################

    k = 1
    p1 = plot()

    # Local source contributions (g_ik) per i=1,...,n
    for i in 1:n
        # Scegli un colore in base al valore di `n_sources` (o `k`)
        plot!(p1, 1:n_time, dat[:g][i, k, 1:n_time], linecolor = :red, lw = 1)
    end

    # Aggiungere global source contribution (f)
    plot!(p1, 1:n_time, dat[:f][k, :], label = "f", linecolor = :black, linewidth = 2)

    # Specifiche grafiche
    xlabel!(p1, "Time")
    ylabel!(p1, "")
    plot!(p1, legend=:false)
    display(p1)
    

    k = 2 
    p2 = plot()

    # Local source contributions (g_ik) per i=1,...,n
    for i in 1:n
        # Scegli un colore in base al valore di `n_sources` (o `k`)
        plot!(p2, 1:n_time, dat[:g][i, k, 1:n_time], linecolor = :red, lw = 1)
    end

    # Aggiungere global source contribution (f)
    plot!(p2, 1:n_time, dat[:f][k, :], label = "f", linecolor = :black, linewidth = 2)

    # Specifiche grafiche
    xlabel!(p2, "Time")
    ylabel!(p2, "")
    plot!(p2, legend=:false)
    display(p2)

   
    return sites, dat, theta_true, dat_trials

end