using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities_new.jl")
include("fit_rpagp.jl")
include("priors_new.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("priors_new.jl")
include("likelihood.jl")


function data_plot()
    # Specifica i parametri del processo MCMC
    n_iter = 1000  # Numero totale di iterazioni
    burn_in = Int(0.6 * n_iter)

    n_time=365

    # Inizializza contenitori per i dati
    chain_f = []
    chain_gamma = []
    chain_tau = []
    chain_rho = []
    chain_g_hat = []
    chain_z = []

    f_true = nothing
    g_true = nothing

    file = h5open("matrici.h5", "r")

    g_true = file["g_true"][:,:,:]
    f_true = file["f_true"][:,:]
    for i in burn_in +1:n_iter
        # Recupera i dati e aggiungili ai rispettivi contenitori
        push!(chain_f, file["f_$i"][:, :])
        push!(chain_gamma, file["gamma_$i"][:])
        push!(chain_tau, file["tau_$i"])
        push!(chain_rho, file["rho_$i"])
        push!(chain_g_hat, file["g_hat_$i"][:, :])
        push!(chain_z, file["z_$i"][:, :])
    end

    close(file)

    # - Estrazione di f
    chain_f_burned = zeros(n_time, n_iter - burn_in)  # Matrice per f
    ss = 1
    for tt in 1:n_iter - burn_in
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
    

        plot!(1:n_time, f_mean, label="Estimated f", linewidth=2, color=:chartreuse)


    display(p12)

end

data_plot()

