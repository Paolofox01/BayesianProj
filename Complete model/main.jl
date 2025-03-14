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
    K = 2  #numero di fonti
    N = 32 #numero di siti
    T = 40 #365
    C = 6 # numero di inquinanti
    
    
    # generating simulated data
    tt, dat, theta_true = simulate_data(df, seed, N, C, T, K) #, dat_trials, y_ict, h

    

    ####################################################################################

    # Iperparametri per MCMC
    hyperparam = Dict(
        # prior
        :tau_prior_sd => sqrt(3), 
        :rho_prior_shape => 0.02, 
        :rho_prior_scale => 1.0,
        :phi_prior_shape => 0.002,
        :phi_prior_scale => 1.0,
        :prior_h_alpha0 => ones(C),
        :prior_sc_a => 3,
        :prior_sc_b => 2,
        #proposal
        :tau_proposal_sd => 0.01,
        :rho_proposal_sd => 0.01, 
        :phi_proposal_sd => 0.01,
        :gamma_proposal_sd => 0.01,
        :h_proposal_sd => 0.01,
        #:g_proposal_sd => 0.01
    )
    
    
    theta0 = Dict{Int64, Dict{Any,Any}}()
    for k in 1:K
        theta0[k] = Dict(
            :rho => 0.3,
            :phi => 1/500.0,
            :beta => zeros(4),
            :tau => zeros(N),
            :gamma => zeros(N),
            :Sigma_gamma => I(N),
            :h =>  ones(C)./C,
            :f => zeros(T),
            :g => zeros(N, T),
            :Sigma_f => I(T),
            :Sigma_f_inv => I(T)
        )
    end

    # Iterazioni di MCMC
    #k=1
    n_iter = 200

    Random.seed!(seed)
    results = fit_model(tt, K, dat, theta0, n_iter, hyperparam)



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
    #MSE_RPAGP = sum((f_hat.med .- dat.f).^2) / T
    #println("MSE(RPAGP): ", MSE_RPAGP)

    # Calcolare l'errore quadratico medio per EMP
    #MSE_EMP = sum((f_EMP.value_mean .- dat.f).^2) / T
    #println("MSE(EMP): ", MSE_EMP)
     p12= plot()


     plot!(dat_trials.time, out_sim.lwr, label="Observed Data", alpha=0.25, linewidth=1)

     plot!(dat_trials.time, out_sim.upr, label="Observed Data", alpha=0.25, linewidth=1)


     plot!(out_sim_summary.time, mean_f, label="Estimated f", linewidth=2, color=:chartreuse)

     plot!(p12, 1:T, dat[:f][1,:], label="Truth", linestyle=:dash, linewidth=2, color=:darkgreen)
  
     display(p12)


     p13 = plot()  # Inizializza il grafico

     # Aggiungi i punti per i diversi set di valori
     scatter!(1:32, theta_true[k][:gamma], label="Gamma True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
     scatter!(1:32, results[:chain][n_iter][k][:gamma], label="mean of sampled Gammas", color=:blue, marker=:circle)
     
     # Mostra il grafico
     display(p13)


     p15 = plot()  # Inizializza il grafico

     # Aggiungi i punti per i diversi set di valori
     scatter!(1:4, theta_true[k][:beta], label="Beta True", xlabel="Posizione", ylabel="Valore", color=:red, marker=:circle)
     scatter!(1:4, results[:chain][n_iter][k][:beta], label="betas estimate", color=:blue, marker=:circle)
     
     # Mostra il grafico
     display(p15)
     

     p14 = plot()  # Inizializza il grafico

     plot!(1:n_iter, [results[:chain][i][1][:beta][4] for i in 1:n_iter])
     
     # Mostra il grafico
     display(p14)

  

end

main()



# TRACEPLOTS
burn_in = Int(0.6* n_iter)  # Calcolare il burn-in (primo 60%)
#burn_in = 1

# select k=1,2
k=1


# betas
p_beta = Plots.Plot{Plots.GRBackend}[]
for num in 1:4
    p_beta_curr = plot()  # Inizializza il grafico
    plot!(burn_in:n_iter, [results[:chain][i][k][:beta][num] for i in burn_in:n_iter])
    #plot!([burn_in,n_iter], [theta_true[k][:beta][num], theta_true[k][:beta][num]])
    push!(p_beta, p_beta_curr)
end
plot(p_beta...)

beta_CI = plot()
plot!(1:4, [median([results[:chain][i][k][:beta][num] for i in burn_in:n_iter]) for num in 1:4], seriestype=:scatter)
plot!(1:4, [quantile([results[:chain][i][k][:beta][num] for i in burn_in:n_iter], 0.975) for num in 1:4], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:4, [quantile([results[:chain][i][k][:beta][num] for i in burn_in:n_iter], 0.025) for num in 1:4], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:4, [theta_true[k][:beta][num] for num in 1:4], seriestype=:scatter, mc=:red)




# rho
p_rho = plot()  # Inizializza il grafico
plot!(1:n_iter, [results[:chain][i][k][:rho] for i in 1:n_iter])
plot!([1,n_iter], [theta_true[k][:rho], theta_true[k][:rho]])



# phi
p_phi = plot()  # Inizializza il grafico
plot!(burn_in:n_iter, [results[:chain][i][k][:phi] for i in burn_in:n_iter])
plot!([burn_in,n_iter], [theta_true[k][:phi], theta_true[k][:phi]])


# gammas
p_gamma = Plots.Plot{Plots.GRBackend}[]
for num in 1:N
    p_gamma_curr = plot()  # Inizializza il grafico
    plot!(burn_in:n_iter, [results[:chain][i][k][:gamma][num] for i in burn_in:n_iter])
    plot!([burn_in,n_iter], [theta_true[k][:gamma][num], theta_true[k][:gamma][num]])
    push!(p_gamma, p_gamma_curr )
end
plot(p_gamma[1:9]...)
plot(p_gamma[10:18]...)
plot(p_gamma[19:27]...)
plot(p_gamma[28:32]...)
plot(p_gamma[1])

p_gamma = plot()
plot!(1:N, [median([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter]) for num in 1:N], seriestype=:scatter)
plot!(1:N, [quantile([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter], 0.975) for num in 1:N] , seriestype=:scatter, mc=:blue, ms=1)
plot!(1:N, [quantile([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter], 0.025) for num in 1:N] , seriestype=:scatter, mc=:blue, ms=1)
plot!(1:N, [theta_true[k][:gamma][num] for num in 1:N] , seriestype=:scatter)




# tau
p_tau = Plots.Plot{Plots.GRBackend}[]
for num in 1:N
    p_tau_curr = plot()  # Inizializza il grafico
    plot!(burn_in:n_iter, [results[:chain][i][k][:tau][num] for i in burn_in:n_iter])
    plot!([burn_in,n_iter], [theta_true[k][:tau][num], theta_true[k][:tau][num]])
    push!(p_tau, p_tau_curr )
end
plot(p_tau[1:9]...)
plot(p_tau[10:18]...)
plot(p_tau[19:27]...)
plot(p_tau[28:32]...)
plot(p_tau[1])



# f
p_f = plot()  # Inizializza il grafico
num = 31
plot!(burn_in:n_iter, [results[:chain][i][k][:f][num] for i in burn_in:n_iter])
plot!([burn_in,n_iter], [theta_true[k][:f][num], theta_true[k][:f][num]])

p_f = plot()  # Inizializza il grafico
plot!(1:T, [median([results[:chain][i][k][:f][num] for i in burn_in:n_iter]) for num in 1:T], linewidth=2, label="f posterior median")
plot!(1:T, [quantile([results[:chain][i][k][:f][num] for i in burn_in:n_iter], 0.025) for num in 1:T], linestyle=:dash, linecolor=:blue,  label="f posterior 95% CIs")
plot!(1:T, [quantile([results[:chain][i][k][:f][num] for i in burn_in:n_iter], 0.975) for num in 1:T], linestyle=:dash, linecolor=:blue, label=missing)
plot!(1:T, theta_true[k][:f], linecolor=:red,  label="true f")

# standardizzo
p_f = plot()  # Inizializza il grafico
f_med = [median([results[:chain][i][k][:f][num] for i in burn_in:n_iter]) for num in 1:T]
plot!(1:T, f_med  ./ norm(f_med ,2))
plot!(1:T, [quantile([results[:chain][i][k][:f][num] for i in burn_in:n_iter], 0.025) for num in 1:T]./ norm(f_med ,2), linestyle=:dash, linecolor=:blue)
plot!(1:T, [quantile([results[:chain][i][k][:f][num] for i in burn_in:n_iter], 0.975) for num in 1:T]./ norm(f_med ,2), linestyle=:dash, linecolor=:blue)
plot!(1:T, theta_true[k][:f]./ norm(theta_true[k][:f] ,2), linecolor=:red)
# gamma std
p_gamma = plot()
plot!(1:N, [median([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter]) for num in 1:N] .+ log(norm(f_med ,2)), seriestype=:scatter, ms=5)
plot!(1:N, [quantile([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter], 0.975) for num in 1:N] .+ log(norm(f_med ,2)), seriestype=:scatter, mc=:blue, ms=2)
plot!(1:N, [quantile([results[:chain][i][k][:gamma][num] for i in burn_in:n_iter], 0.025) for num in 1:N] .+ log(norm(f_med ,2)), seriestype=:scatter, mc=:blue, ms=2)
plot!(1:N, [theta_true[k][:gamma][num] for num in 1:N] .+ log(norm(theta_true[k][:f] ,2)), seriestype=:scatter, mc=:red, ms=3)



# g 
p_g = Plots.Plot{Plots.GRBackend}[]
for staz in 1:N
    p_g_curr = plot()
    plot!(1:T, [median([results[:chain][i][k][:g][staz, num] for i in burn_in:n_iter]) for num in 1:T], label="g_$staz posterior median", linewidth=2)
    plot!(1:T, [quantile([results[:chain][i][k][:g][staz, num] for i in burn_in:n_iter], 0.025) for num in 1:T], linestyle=:dash, linecolor=:blue, label="g_$staz posterior 95% CIs")
    plot!(1:T, [quantile([results[:chain][i][k][:g][staz, num] for i in burn_in:n_iter], 0.975) for num in 1:T], linestyle=:dash, linecolor=:blue, label=missing)
    plot!(1:T, theta_true[k][:g][staz,:], linecolor=:red, label="true g_$staz")
    push!(p_g, p_g_curr)
end

plot(p_g[1:9]...)
plot(p_g[10:18]...)
plot(p_g[19:26]...)
plot(p_g[20])




# h
p_h = Plots.Plot{Plots.GRBackend}[]
for num in 1:C
    p_h_curr = plot()  # Inizializza il grafico
    plot!(burn_in:n_iter, [results[:chain][i][k][:h][num] for i in burn_in:n_iter])
    plot!([burn_in,n_iter], [theta_true[k][:h][num], theta_true[k][:h][num]])
    push!(p_h, p_h_curr)
end
plot(p_h...)

h_CI = plot()
plot!(1:C, [median([results[:chain][i][k][:h][num] for i in burn_in:n_iter]) for num in 1:C], seriestype=:scatter)
plot!(1:C, [quantile([results[:chain][i][k][:h][num] for i in burn_in:n_iter], 0.975) for num in 1:C], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:C, [quantile([results[:chain][i][k][:h][num] for i in burn_in:n_iter], 0.025) for num in 1:C], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:C, [theta_true[k][:h][num] for num in 1:C], seriestype=:scatter, mc=:red)


# sigma_c
p_sigma = Plots.Plot{Plots.GRBackend}[]
for num in 1:C
    p_sigma_curr = plot()  # Inizializza il grafico
    plot!(burn_in:n_iter, [results[:chain_sigma2_c][i][num] for i in burn_in:n_iter])
    plot!([burn_in,n_iter], [dat[:sigma2_c][num],dat[:sigma2_c][num]])
    push!(p_sigma, p_sigma_curr)
end
plot(p_sigma...)

sigma_CI = plot()
plot!(1:C, [median([results[:chain_sigma2_c][i][num] for i in burn_in:n_iter]) for num in 1:C], seriestype=:scatter)
plot!(1:C, [quantile([results[:chain_sigma2_c][i][num] for i in burn_in:n_iter], 0.975) for num in 1:C], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:C, [quantile([results[:chain_sigma2_c][i][num] for i in burn_in:n_iter], 0.025) for num in 1:C], seriestype=:scatter, mc=:blue, ms=2)
plot!(1:C, [sqrt(dat[:sigma2_c][num]) for num in 1:C], seriestype=:scatter, mc=:red)



# y
staz = 10 #selct station 1:32
p_y = Plots.Plot{Plots.GRBackend}[]
for comp in 1:C
    p_y_curr = plot()
    plot!(1:T, [median([results[:chain_y][i][staz, comp, num] for i in burn_in:n_iter]) for num in 1:T], label="y_$comp median", linewidth=2)
    plot!(1:T, [quantile([results[:chain_y][i][staz, comp, num] for i in burn_in:n_iter], 0.025) for num in 1:T], linestyle=:dash, linecolor=:blue, label="y_$comp 95% CIs")
    plot!(1:T, [quantile([results[:chain_y][i][staz, comp, num] for i in burn_in:n_iter], 0.975) for num in 1:T], linestyle=:dash, linecolor=:blue, label=missing)
    plot!(1:T, dat[:y][staz, comp, :], linecolor=:red, label="true y_$comp")
    push!(p_y, p_y_curr)
end

plot(p_y...)
plot(p_y[5])