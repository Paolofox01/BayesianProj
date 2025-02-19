
using Random, Plots, DataFrames, StatsBase, Distributions, ToeplitzMatrices, CSV, HDF5

include("utilities.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")


function simulate_data(df, seed, N, C, T, K)
    # Impostazione del seed
    Random.seed!(seed)

    intercept = ones(N)
    x1 = zeros(N)
    x2 = zeros(N)

    for cont in 1:N
        if df[cont, 8] == "SUBURBAN"
            x1[cont] = 1
        end
        
        if df[cont, 8] == "URBAN AND CENTER CITY"
            x2[cont] = 1
        end

    end

    # Design matrix: sites' territorial characterization
    X = Matrix(DataFrame(intercept=intercept, suburban=x1, urban=x2, ElevationSTD = (df[:, 3].- mean(df[:, 3])) ./ std(df[:, 3])))
    # Geographical coordinates: latitude and longitude
    coords = Matrix(DataFrame(Latitude = df[:, 1], Longitude = df[:, 2]))  
    maximum(euclid_dist(coords[: , 1], coords[: , 2], N))


    # Parametri del modello
    theta = Dict{Int64, Dict{Any,Any}}()
    theta[1] = Dict(
        :rho => 0.15,
        :phi => 1/300.0,
        :beta => [-0.6, 0.2, 0.5, -0.3],
        :tau => rand(Normal(0, 1), N),
        :gamma => zeros(N),
        :h =>  ones(C)./C,
        :f => zeros(T),
        :g => zeros(N, T),
        :sigma2_c => zeros(C)
    )

    theta[2] = Dict(
        :rho => 0.25,
        :phi => 1/400.0,
        :beta => [-0.2, 0.4, -0.6, 0.1],
        :tau => rand(Normal(0, 1), N),
        :gamma => zeros(N),
        :h =>  ones(C)./C,
        :f => zeros(T),
        :g => zeros(N, T),
        :sigma2_c => zeros(C)
    )

    # Funzione per generare i dati (devi definire `generate_data` in Julia)
    dat, theta_true = generate_data(X, coords, theta, N, C, T, K)



    # salvo i dati simulati per ogni k=1,..,K
    # df_new = Dict{Int64, DataFrame}()
    # dat_trials = Dict{Int64, DataFrame}()
    # for k in 1:K
    #     theta_true[k][:gamma] = dat[:gamma][:,k]
    #     df_new[k] = DataFrame(dat[:g][:, k, :], :auto)
    #     # "Melt" della matrice (equivalente a reshape2::melt in R)
    #     dat_trials[k] = stack(df_new[k], variable_name = "time", value_name = "value")
    #     # Aggiungiamo una colonna per il trial
    #     dat_trials[k].trial = repeat(1:n,n_time)
    # end








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

    for c in 1:C
        @eval $(Symbol(:pyc, c)) = plot()
        for i in 1:n
            # Scegli un colore in base al valore di `n_sources` (o `k`)
            plot!((@eval $(Symbol(:pyc, c))), 1:n_time, y_ict[i, c, 1:n_time], linecolor = :black, lw = 1)
        end
        # Specifiche grafiche
        xlabel!((@eval $(Symbol(:pyc, c))), "Time")
        ylabel!((@eval $(Symbol(:pyc, c))), "")
        plot!((@eval $(Symbol(:pyc, c))), legend=:false)
        display(@eval $(Symbol(:pyc, c)))
    end
    
   
    return sites, dat, theta_true, dat_trials, y_ict, h

end