using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities.jl")
include("fit_MCMC.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")
include("simulated_data.jl")


function main()
    # Set a fixed seed
    seed = 1997

    df = CSV.read("./FinalStations.csv", DataFrame; delim=',')


    # Parametri
    K = 2 #numero di fonti
    n = 32 #numero di siti
    n_time = 30 #365
    C = 6 # numero di inquinanti
    
    # TODO: standardize altitude!!
    
    # generating simulated data
    sites, dat, theta_true, dat_trials = simulate_data(df, seed, K, n, C, n_time)

    

    ####################################################################################

    # Iperparametri per MCMC
    hyperparam = Dict(
        :tau_prior_sd => sqrt(3), 
        :tau_proposal_sd => 0.01,
        :rho_prior_shape => 0.02, 
        :rho_prior_scale => 1,
        :rho_proposal_sd => 0.01, 
        #:beta_prior_mu => 0, 
        #:beta_prior_sd => 1
        :phi_prior_shape => 0.02,
        :phi_prior_scale => 1.0,
        :phi_proposal_sd => 0.05,
        :beta_proposal_sd => 0.1,
        :gamma_proposal_sd => 0.01
    )
    
    #theta0 = Dict{Int64, Dict{Any,Any}}()
    theta0 = Dict(
        :rho => 0.3,
        :phi => 1/500.0,
        :beta => ones(4),
        :gamma => (ones(n)), 
        :tau => zeros(n) 
    )


   
    # Punto e valore fissati (pinned point/value)
    #pinned_point = div(n_time, 2)  # punto fissato (metà del tempo)
   # pinned_value = mean(dat[:g][:, 1, pinned_point]) # valore medio della colonna `pinned_point` (in R 'apply(dat$y, 1, mean)[pinned_point]')

    # Iterazioni di MCMC
    k = 2
    n_iter = 300

    Random.seed!(seed)
    results = fit_model(sites, dat[:g][:,k,:], n_iter, theta0, hyperparam)

    # Funzione per riassumere i risultati MCMC
    burn_in = Int(0.6* n_iter)  # Calcolare il burn-in (primo 60%)

     # Salvare tutte le matrici in un unico file HDF5
     h5open("matrici.h5", "w") do file
        file["rho_true"] = theta_true[k][:rho]
        file["phi_true"] = theta_true[k][:phi]
        file["beta_true"] = theta_true[k][:beta]
        file["tau_true"] = theta_true[k][:tau]
        file["gamma_true"] = dat[:gamma]
        file["g_true"] = dat[:g]
        file["f_true"] = dat[:f]
        for i in 1:n_iter
            file["f_$i"] = results[:chain_f][i]
            file["gamma_$i"] = results[:chain][i][:gamma]
            file["tau_$i"] = results[:chain][i][:tau]
            file["rho_$i"] = results[:chain][i][:rho]
            file["g_hat_$i"] = results[:chain_g][i]
            file["z_$i"] = results[:chain_z][i]
            file["beta_$i"] = results[:chain][i][:beta]
            file["phi_$i"] = results[:chain][i][:phi]
        end
    end

    out_sim, mean_f, g_hat_quantiles = getSummaryOutput(results, dat_trials[k], dat[:g][:,k,:], burn_in)

    #######################################################################################

    # Supponiamo che `dat_trials` e `out_sim` siano già in formato DataFrame o simile

    # Calcolare la mediana dei valori previsti
    out_sim_summary = combine(groupby(out_sim, :time), :med => median)




    # DA QUI: sistemare parte di posterior inference per display risultati





    # Calcolare la media empirica dei dati
    dat_trials_summary = combine(groupby(dat_trials, :time), :value => mean)

    # Crea il grafico
    plot()

    # Linee per i dati osservati (trials)
    #plot!(dat_trials.time, out_sim.lwr, label="Observed Data", alpha=0.25, linewidth=1)

    #plot!(dat_trials.time, out_sim.upr, label="Observed Data", alpha=0.25, linewidth=1)

    # Linea per il valore stimato della funzione f
    plot!(out_sim_summary.time, out_sim_summary.med_median, label="median f", linewidth=2, color=:chartreuse)

    plot!(out_sim_summary.time, mean_f, label="mean f", linewidth=2, color=:chartreuse)

    
    plot!(out_sim_summary.time, dat[:f][k,:], label="True f", linewidth=2, color=:red)

    # Linea per la media empirica
    #plot!(dat_trials_summary.time, dat_trials_summary.value_mean, label="Empirical Mean", linestyle=:dot, linewidth=2, color=:black)

    ############################################################################

    # Calcolare la mediana di f_hat (come in R)
    #f_hat = combine(groupby(out_sim, :time), :med => median)

    # Calcolare la media empirica di f_EMP (come in R)
    #f_EMP = combine(groupby(dat_trials, :time), :value => mean)

    # Calcolare l'errore quadratico medio per RPAGP
    #MSE_RPAGP = sum((f_hat.med .- dat.f).^2) / n_time
    #println("MSE(RPAGP): ", MSE_RPAGP)

    # Calcolare l'errore quadratico medio per EMP
    #MSE_EMP = sum((f_EMP.value_mean .- dat.f).^2) / n_time
    #println("MSE(EMP): ", MSE_EMP)
     p12= plot()


     plot!(dat_trials.time, out_sim.lwr, label="Observed Data", alpha=0.25, linewidth=1)

     plot!(dat_trials.time, out_sim.upr, label="Observed Data", alpha=0.25, linewidth=1)


     plot!(out_sim_summary.time, mean_f, label="Estimated f", linewidth=2, color=:chartreuse)

     plot!(p12, 1:n_time, dat[:f][1,:], label="Truth", linestyle=:dash, linewidth=2, color=:darkgreen)
  
     display(p12)


     p13 = plot()  # Inizializza il grafico

     # Aggiungi i punti per i diversi set di valori
     scatter!(1:32, theta_true[k][:gamma], label="Gamma True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
     scatter!(1:32, results[:chain][n_iter][:gamma], label="mean of sampled Gammas", color=:blue, marker=:circle)
     
     # Mostra il grafico
     display(p13)


     p15 = plot()  # Inizializza il grafico

     # Aggiungi i punti per i diversi set di valori
     scatter!(1:4, theta_true[k][:beta], label="Beta True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
     scatter!(1:4, results[:chain][n_iter][:beta], label="betas estimate", color=:blue, marker=:circle)
     
     # Mostra il grafico
     display(p15)
     

     p14 = plot()  # Inizializza il grafico

     plot!(1:n_iter, [results[:chain][i][:beta][4] for i in 1:n_iter])
     
     # Mostra il grafico
     display(p14)

  

end

main()



