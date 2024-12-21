using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities_new.jl")
include("fit_rpagp.jl")
include("priors_new.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("priors_new.jl")
include("likelihood.jl")

function main()
    # Impostazione del seed
    

    df = CSV.read("./FinalStations.csv", DataFrame; delim=',')

    Random.seed!(1008)

    # Parametri
    K = 2 #numero di fonti
    n = 32 #numero di siti
    n_time = 365
    n_sims = 1


    interc = ones(n)
    x1 = zeros(n)
    x2= zeros(n)

    for cont in 1:n 
        if df[cont, 8] == "SUBURBAN"
            x1[cont] = 1
        end
        
        if df[cont, 8] == "URBAN AND CENTER CITY"
            x2[cont] = 1
        end

    end


    sites = Matrix(DataFrame(Latitude = df[:, 1], Longitude = df[:, 2], interc = interc, x1 = x1, x2=x2, Elevation = df[:, 3]))
    
    maximum(euclid_dist(sites[: , 1], sites[: , 2], n))

    # Parametri del modello
    theta = Dict(
        :rho_f => 0.1,
        :rho => [300.0, 400.0],
        :beta => [0.5 0.5 1. -0.1; 0.3 -0.4 -0.7 -0.02],
        :tau => rand(Normal(0, 3), n) 
    )



    # Funzione per generare i dati (devi definire `generate_data` in Julia)
    dat = generate_data(sites, n, K, n_time, theta)

    # Creiamo un DataFrame dalla matrice
    df_new = DataFrame(dat[:g][:, 1, :], :auto)

    # "Melt" della matrice (equivalente a reshape2::melt in R)
    dat_trials = stack(df_new, variable_name = "time", value_name = "value")

    # Aggiungiamo una colonna per il trial
    dat_trials.trial = repeat(1:n,n_time)


    ##########################################################################################
    Random.seed!(1008)
    # Creare il grafico
    p1 = plot()

    # colors = [:blue, :red]  # Aggiungi più colori se necessario

    # # Aggiungi le linee per ogni combinazione di `trial` e `n_sources`
    # for n_sources in 1:K
    #     for trial in 1:n
    #         # Scegli un colore in base al valore di `n_sources` (o `k`)
    #         color_index = (n_sources - 1) % length(colors) + 1  # Ciclo sui colori
    #         plot!(p, 1:n_time, dat[:y][trial, n_sources, 1:n_time], 
    #               linecolor = colors[color_index], lw = 3)
    #     end
    # end


    for trial in 1:n
        # Scegli un colore in base al valore di `n_sources` (o `k`)

        plot!(p1, 1:n_time, dat[:g][trial, 1, 1:n_time], 
            linecolor = :red, lw = 1)
    end

    #plot!(p, 1:n_time, dat[:y][9, 1, 1:n_time], linecolor = :red, lw = 3)
    
    # Aggiungere la verità (f)
    plot!(p1, 1:n_time, dat[:f][1, :], label = "f", linecolor = :green, linewidth = 1)

    # Modificare l'asse delle x
    xlabel!(p1, "Time")
    ylabel!(p1, "")

    # Personalizzare i colori, i tipi di linea e altre impostazioni
    # In Julia non c'è una funzione diretta per definire una legenda complessa come in ggplot2,
    # ma possiamo modificare l'etichetta per rappresentare la legenda
    # Se vuoi personalizzare ulteriormente la legenda, è necessario lavorare con `guide` in Plots.jl.
    # Plots.jl in Julia non ha un equivalente diretto di `scale_color_manual`, ma puoi modificare la 
    # legenda e i colori in modo simile.

    # Personalizzare la griglia e le etichette degli assi
    plot!(p1, legend=:false)

    #posso mettere top right altriment

    # Mostrare il grafico
    display(p1)


    p2 = plot()

    # colors = [:blue, :red]  # Aggiungi più colori se necessario

    # # Aggiungi le linee per ogni combinazione di `trial` e `n_sources`
    # for n_sources in 1:K
    #     for trial in 1:n
    #         # Scegli un colore in base al valore di `n_sources` (o `k`)
    #         color_index = (n_sources - 1) % length(colors) + 1  # Ciclo sui colori
    #         plot!(p, 1:n_time, dat[:y][trial, n_sources, 1:n_time], 
    #               linecolor = colors[color_index], lw = 3)
    #     end
    # end


    for trial in 1:n
        # Scegli un colore in base al valore di `n_sources` (o `k`)

        plot!(p2, 1:n_time, dat[:g][trial, 2, 1:n_time], 
            linecolor = :red, lw = 1)
    end

    #plot!(p, 1:n_time, dat[:y][9, 1, 1:n_time], linecolor = :red, lw = 3)
    
    # Aggiungere la verità (f)
    plot!(p2, 1:n_time, dat[:f][2, :], label = "f", linecolor = :green, linewidth = 1)

    # Modificare l'asse delle x
    xlabel!(p2, "Time")
    ylabel!(p2, "")

    # Personalizzare i colori, i tipi di linea e altre impostazioni
    # In Julia non c'è una funzione diretta per definire una legenda complessa come in ggplot2,
    # ma possiamo modificare l'etichetta per rappresentare la legenda
    # Se vuoi personalizzare ulteriormente la legenda, è necessario lavorare con `guide` in Plots.jl.
    # Plots.jl in Julia non ha un equivalente diretto di `scale_color_manual`, ma puoi modificare la 
    # legenda e i colori in modo simile.

    # Personalizzare la griglia e le etichette degli assi
    plot!(p2, legend=:false)

    #posso mettere top right altriment

    # Mostrare il grafico
    display(p2)

    #####################################################################################

    # Iperparametri per MCMC
    hyperparam = Dict(
        :tau_prior_sd => sqrt(3), 
        :tau_proposal_sd => 0.1,
        :rho_prior_shape => 5, 
        :rho_prior_scale => 0.02,
        :rho_proposal_sd => 0.05, 
        #:beta_prior_mu => 0, 
        #:beta_prior_sd => 1
        :rho_spatial_prior_shape => 3.0,
        :rho_spatial_prior_scale => 1000.0,
        :rho_spatial_proposal_sd => 1.0,
        :beta_proposal_sd => 0.04,
        :gamma_proposal_sd => 0.03
    )

    theta_true = Dict(
        :rho => 0.1,
        #:rho_spatial => theta[:rho]
        :gamma => dat[:gamma][:, 1],
    )

    theta0 = Dict(
        :rho => 0.4,
        :rho_spatial => 500.0,
        :beta => [0.5, 0.5, 1., -0.1],
        :gamma => (ones(n)), 
        :tau => zeros(n)
    )

    # Punto e valore fissati (pinned point/value)
    pinned_point = div(n_time, 2)  # punto fissato (metà del tempo)
    pinned_value = mean(dat[:g][:, 1, pinned_point]) # valore medio della colonna `pinned_point` (in R 'apply(dat$y, 1, mean)[pinned_point]')

    # Iterazioni di MCMC
    n_iter = 3000
    results = fit_rpagp(sites, dat[:g][:,1,:], n_iter, theta0, hyperparam, pinned_point, pinned_value)

    # Funzione per riassumere i risultati MCMC
    burn_in = Int(0.6 * n_iter)  # Calcolare il burn-in (primo 60%)

     # Salvare tutte le matrici in un unico file HDF5
     h5open("matrici.h5", "w") do file
        file["rho_true"] = theta[:rho_f]
        file["rho_spatial_true"] = theta[:rho]
        file["beta_true"] = theta[:beta]
        file["tau_true"] = theta[:tau]
        file["gamma_true"] = dat[:gamma]
        file["g_true"] = dat[:g]
        file["f_true"] = dat[:f]
        for i in 1:n_iter
            file["f_$i"] = results[:chain_f][i]
            file["gamma_$i"] = results[:chain][i][:gamma]
            file["tau_$i"] = results[:chain][i][:tau]
            file["rho_$i"] = results[:chain][i][:rho]
            file["g_hat_$i"] = results[:chain_g_hat][i]
            file["z_$i"] = results[:chain_z][i]
            file["beta_$i"] = results[:chain][i][:beta]
            file["rho_spatial_$i"] = results[:chain][i][:rho_spatial]
        end
    end

    out_sim, mean_f = getSummaryOutput(results, dat_trials, dat[:g][:,1,:], burn_in)

    #######################################################################################

    # Supponiamo che `dat_trials` e `out_sim` siano già in formato DataFrame o simile

    # Calcolare la mediana dei valori previsti
    out_sim_summary = combine(groupby(out_sim, :time), :med => median)

    # Calcolare la media empirica dei dati
    dat_trials_summary = combine(groupby(dat_trials, :time), :value => mean)

    # Crea il grafico
    plot()

    # Linee per i dati osservati (trials)
    #plot!(dat_trials.time, out_sim.lwr, label="Observed Data", alpha=0.25, linewidth=1)

    #plot!(dat_trials.time, out_sim.upr, label="Observed Data", alpha=0.25, linewidth=1)

    # Linea per il valore stimato della funzione f
    plot!(out_sim_summary.time, out_sim_summary.med_median, label="Estimated f", linewidth=2, color=:chartreuse)

    plot!(out_sim_summary.time, mean_f, label="Estimated f", linewidth=2, color=:chartreuse)

    
    #plot!(out_sim_summary.time,sample_f(dat[:g][:,1,:], theta0, 1), label="Estimated f", linewidth=2, color=:chartreuse)

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
     scatter!(1:32, theta_true[:gamma], label="Gamma True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
     scatter!(1:32, results[:chain][n_iter][:gamma], label="mean of sampled Gammas", color=:blue, marker=:circle)
     
     # Mostra il grafico
     display(p13)


     

     p14 = plot()  # Inizializza il grafico

     plot!(1:n_iter, [results[:chain][i][:rho] for i in 1:n_iter])
     
     # Mostra il grafico
     display(p14)

  

end

main()
