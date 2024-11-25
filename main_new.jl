using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV




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
    df_new = DataFrame(dat[:g][:, 2, :], :auto)

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
        :tau_prior_sd => 4, 
        :tau_proposal_sd => 0.1,
        :rho_prior_shape => 5, 
        :rho_prior_scale => 0.01,
        :rho_proposal_sd => 0.01, 
        :beta_prior_mu => 1, 
        :beta_prior_sd => 0.5
    )

    theta_true = Dict(
        :rho => 0.1,
        :beta => dat[:gamma][:, 1],
        :tau => rand(Normal(0, 3), n)
    )

    theta0 = Dict(
        :rho => 0.1,
        :beta => exp.(ones(n)), 
        :tau => zeros(n)
    )

    # Punto e valore fissati (pinned point/value)
    pinned_point = div(n_time, 2)  # punto fissato (metà del tempo)
    pinned_value = mean(dat[:g][:, 1, pinned_point]) # valore medio della colonna `pinned_point` (in R 'apply(dat$y, 1, mean)[pinned_point]')

    # Iterazioni di MCMC
    n_iter = 3000
    results = fit_rpagp(dat[:g][:,1,:], n_iter, theta0, hyperparam, pinned_point, pinned_value)

    # Funzione per riassumere i risultati MCMC
    burn_in = Int(0.6 * n_iter)  # Calcolare il burn-in (primo 60%)
    out_sim = getSummaryOutput(results, DataFrame(dat), dat[:g], burn_in)

    #######################################################################################

    # Supponiamo che `dat_trials` e `out_sim` siano già in formato DataFrame o simile

    # Calcolare la mediana dei valori previsti
    out_sim_summary = combine(groupby(out_sim, :time), :med => median)

    # Calcolare la media empirica dei dati
    dat_trials_summary = combine(groupby(dat_trials, :time), :value => mean)

    # Crea il grafico
    plot()

    # Linee per i dati osservati (trials)
    plot!(dat_trials.time, dat_trials.value, label="Observed Data", alpha=0.25, linewidth=1)

    # Linea per il valore stimato della funzione f
    plot!(out_sim_summary.time, out_sim_summary.med, label="Estimated f", linewidth=2, color=:chartreuse)

    # Linea per il valore della "verità" f
    plot!(1:n_time, dat.f, label="Truth", linestyle=:dash, linewidth=2, color=:darkgreen)

    # Linea per la media empirica
    plot!(dat_trials_summary.time, dat_trials_summary.value_mean, label="Empirical Mean", linestyle=:dot, linewidth=2, color=:black)

    # Impostazioni finali del grafico
    xlabel!("Time")
    ylabel!("")

    # Personalizzare la legenda
    plot!(label=["Weekly Forecast", "Main Forecast", "ciao"], 
        linewidth=[1, 2, 2], 
        linestyle=[:dot, :solid, :dash],
        color=[:red, :chartreuse, :darkgreen])

    # Impostazioni finali per il tema
    plot!(grid=false, ticks=false, legend=:topright)

    ############################################################################

    # Calcolare la mediana di f_hat (come in R)
    f_hat = combine(groupby(out_sim, :time), :med => median)

    # Calcolare la media empirica di f_EMP (come in R)
    f_EMP = combine(groupby(dat_trials, :time), :value => mean)

    # Calcolare l'errore quadratico medio per RPAGP
    MSE_RPAGP = sum((f_hat.med .- dat.f).^2) / n_time
    println("MSE(RPAGP): ", MSE_RPAGP)

    # Calcolare l'errore quadratico medio per EMP
    MSE_EMP = sum((f_EMP.value_mean .- dat.f).^2) / n_time
    println("MSE(EMP): ", MSE_EMP)

end


main()