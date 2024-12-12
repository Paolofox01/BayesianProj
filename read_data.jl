using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities_new.jl")
include("fit_rpagp.jl")
include("priors_new.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("priors_new.jl")
include("likelihood.jl")
include("traceplot.jl")


function data_plot()
    # Specifica i parametri del processo MCMC
    n_iter = 10000  # Numero totale di iterazioni
    burn_in = Int(0.75 * n_iter)


    # Inizializza contenitori per i dati
    chain_f = []
    chain_gamma = []
    chain_tau = []
    chain_rho = []
    chain_g_hat = []
    chain_z = []
    chain_beta = []
    chain_rho_spatial = []


    f_true = nothing
    g_true = nothing

        # Inizializza i contenitori al di fuori del blocco `do`
    chain_f = []
    chain_gamma = []
    chain_tau = []
    chain_rho = []
    chain_g_hat = []
    chain_z = []
    chain_beta = []
    chain_rho_spatial = []

    # Inizializza g_true e f_true all'esterno
    g_true = nothing
    f_true = nothing
    gamma_true = nothing
    beta_true = nothing

    # Apri il file HDF5 e leggi i dati
    h5open("matrici.h5", "r") do file
        # Assegna i dati statici
        g_true = copy(file["g_true"][:, :, :])
        f_true = copy(file["f_true"][:, :])
        gamma_true = copy(file["gamma_true"][:, :])
        beta_true = copy(file["beta_true"][:,:])

        # Itera sugli indici desiderati
        for i in 1:n_iter
            # Aggiungi i dati ai contenitori
            push!(chain_f, copy(file["f_$i"][:, :]))
            push!(chain_gamma, copy(file["gamma_$i"][:]))
            push!(chain_tau, copy(file["tau_$i"][:]))
            push!(chain_rho, copy(file["rho_$i"][]))
            push!(chain_g_hat, copy(file["g_hat_$i"][:, :]))
            push!(chain_z, copy(file["z_$i"][:, :]))
            push!(chain_beta, copy(file["beta_$i"][:]))
            push!(chain_rho_spatial, copy(file["rho_spatial_$i"][]))
        end
    end

    n_time = length(chain_f[1])
    n = length(chain_gamma[1])
    n_covariate = length(chain_beta[1])
        

    # - Estrazione di f
    chain_f_burned = zeros(n_time, n_iter - burn_in)  # Matrice per f
    ss = 1
    for tt in burn_in+1:n_iter
        chain_f_burned[:, ss] = chain_f[tt]  # f dalla catena
        ss += 1
    end

    # Definisci i quantili
    probs = [0.025, 0.5, 0.975]
        
    # Inizializza un array per i quantili
    f_hat_quantiles = Array{Float64}(undef, 3, n_time)
    
    # Calcola i quantili per ogni time point e trial

    for t in 1:n_time
            f_hat_quantiles[:, t] = quantile(chain_f_burned[t, :], probs)
    end

    f_mean = vec(mean(f_hat_quantiles, dims = 1))


    p12= plot()


        plot!(1:n_time, f_hat_quantiles[1, :], label="Observed Data", alpha=0.25, linewidth=1)

        plot!(1:n_time, f_hat_quantiles[3, :], label="Observed Data", alpha=0.25, linewidth=1)


        plot!(1:n_time, f_hat_quantiles[2, :], label="Estimated f", linewidth=2, color=:chartreuse)

        plot!(p12, 1:n_time, f_true[1, :], label="Truth", linestyle=:dash, linewidth=2, color=:darkgreen)
    

        #plot!(1:n_time, f_mean, label="Estimated f", linewidth=2, color=:chartreuse)


    display(p12)

    chains = [chain_gamma, chain_tau, chain_rho, chain_beta, chain_rho_spatial]
    chain_names = ["Gamma", "Tau", "Rho", "Beta", "Rho_spatial"]

    for (chain, name) in zip(chains, chain_names)
        traceplot(chain, name, burn_in)
    end

    gamma_burned = zeros(n, n_iter - burn_in)  # Matrice per f
    ss = 1
    for tt in burn_in+1:n_iter
        gamma_burned[:, ss] = chain_gamma[tt]  # f dalla catena
        ss += 1
    end

    # Inizializza un array per i quantili
    gamma_quantiles = Array{Float64}(undef, 3, n)

    for k in 1:n
        gamma_quantiles[:, k] = quantile(gamma_burned[k, :], probs)
    end

    gamma_error_bars = (gamma_quantiles[2, :] - gamma_quantiles[1, :], gamma_quantiles[3, :] - gamma_quantiles[2, :])

    p13 = plot()  # Inizializza il grafico

        # Aggiungi i punti per i diversi set di valori
        scatter!(1:32, gamma_true[:, 1], label="Gamma True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
        scatter!(1:32, gamma_quantiles[2,:], yerror=gamma_error_bars, label="sampled Gammas", color=:blue, marker=:circle)
     
     # Mostra il grafico
    display(p13)

    beta_burned = zeros(n_covariate, n_iter - burn_in)  # Matrice per f
    ss = 1
    for tt in burn_in+1:n_iter
        beta_burned[:, ss] = chain_beta[tt]  # f dalla catena
        ss += 1
    end

    # Inizializza un array per i quantili
    beta_quantiles = Array{Float64}(undef, 3, n_covariate)

    for covv in 1:n_covariate
        beta_quantiles[:, covv] = quantile(beta_burned[covv, :], probs)
    end

    beta_error_bars = (beta_quantiles[2, :] - beta_quantiles[1, :], beta_quantiles[3, :] - beta_quantiles[2, :])

    p14 = plot()  # Inizializza il grafico

        # Aggiungi i punti per i diversi set di valori
        scatter!(1:4, beta_true[1, :], label="Beta True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
        scatter!(1:4, beta_quantiles[2,:], yerror=beta_error_bars, label="sampled Betas", color=:blue, marker=:circle)
     


     # Mostra il grafico
    display(p14)

    # df = CSV.read("./FinalStations.csv", DataFrame; delim=',')


    # interc = ones(n)
    # x1 = zeros(n)
    # x2= zeros(n)

    # for cont in 1:n 
    #     if df[cont, 8] == "SUBURBAN"
    #         x1[cont] = 1
    #     end
        
    #     if df[cont, 8] == "URBAN AND CENTER CITY"
    #         x2[cont] = 1
    #     end

    # end


    # sites = Matrix(DataFrame(Latitude = df[:, 1], Longitude = df[:, 2], interc = interc, x1 = x1, x2=x2, Elevation = df[:, 3]))
    

    # p15 = plot()  # Inizializza il grafico

    #     # Aggiungi i punti per i diversi set di valori
    #     scatter!(1:32, gamma_true[1, :], label="Beta True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
    #     scatter!(1:32, sites[:, 3:6] * beta_quantiles[2,:], label="sampled Betas", color=:blue, marker=:circle)
     


    #  # Mostra il grafico
    # display(p15)


    # p15 = plot()  # Inizializza il grafico

    # histogram!(chain_rho)

    # # Mostra il grafico
    # display(p15)

end

data_plot()

