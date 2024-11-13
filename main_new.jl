using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices

include("fit_rpagp.jl")
include("likelihood.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("utilities_new.jl")

# Impostazione del seed
Random.seed!(1008)



# Parametri
K = 2 #numero di fonti
n = 20 #numero di siti
n_time = 365
n_sims = 1



sites = hcat(
    10.0 .+ 13.0 * rand(n),  # Colonna 1 (valori reali)
    10.0 .+ 35.0 * rand(n),  # Colonna 2 (valori reali)
    rand(Bool, n),           # Colonna 3 (binaria)
    rand(Bool, n),           # Colonna 4 (binaria)
    rand(Bool, n),           # Colonna 5 (binaria)
    1.0 .+ 20.0 .* rand(n)     # Colonna 6 (valori reali)
)
# Parametri del modello
theta = Dict(
    :rho_f => 4.0,
    :rho => [3.5, 2.0],
    :beta => [1.1 -0.5 0.4 -0.5; 0.3 -0.4 0.7 0.2],
    :tau =>  3 .* rand(n) ,
)


# Funzione per generare i dati (devi definire `generate_data` in Julia)
dat = generate_data(sites, n, K, n_time, theta)

# Reshaping dei dati (come in R con reshape2::melt)
# dat_trials = reshape(dat[:y], n_time, n, n_sims)  # Modifica questo in base alla struttura effettiva dei dati

##########################################################################################
# Creare il grafico
p = plot()

colors = [:blue, :red]  # Aggiungi più colori se necessario

# Aggiungi le linee per ogni combinazione di `trial` e `n_sources`
for n_sources in 1:K
    for trial in 1:n
        # Scegli un colore in base al valore di `n_sources` (o `k`)
        color_index = (n_sources - 1) % length(colors) + 1  # Ciclo sui colori
        plot!(p, 1:n_time, dat[:y][trial, n_sources, 1:n_time], 
              linecolor = colors[color_index], lw = 3)
    end
end

# Aggiungere la verità (f)
plot!(p, 1:n_time, dat[:f]', label = "Truth", linecolor = :green, linewidth = 2)

# Modificare l'asse delle x
xlabel!(p, "Time")
ylabel!(p, "")

# Personalizzare i colori, i tipi di linea e altre impostazioni
# In Julia non c'è una funzione diretta per definire una legenda complessa come in ggplot2,
# ma possiamo modificare l'etichetta per rappresentare la legenda
# Se vuoi personalizzare ulteriormente la legenda, è necessario lavorare con `guide` in Plots.jl.
# Plots.jl in Julia non ha un equivalente diretto di `scale_color_manual`, ma puoi modificare la 
# legenda e i colori in modo simile.

# Personalizzare la griglia e le etichette degli assi
plot!(p, legend=:false, grid = false, axis = false, frame = false)

#posso mettere top right altriment

# Mostrare il grafico
display(p)

#####################################################################################

# Iperparametri per MCMC
hyperparam = Dict(
    :tau_prior_sd => 0.2, 
    :tau_proposal_sd => 1e-3,
    :rho_prior_shape => 12, 
    :rho_prior_scale => 1,
    :rho_proposal_sd => 1, 
    :beta_prior_mu => 1, 
    :beta_prior_sd => 0.5
)

# Valori iniziali
n = 20  # Numero di osservazioni (aggiustato per il tuo esempio)
n_time = 50  # Numero di tempi (aggiustato per il tuo esempio)

theta0 = Dict(
    :rho => 15,
    :beta => ones(n),  # Creazione di un array di n elementi con valore 1
    :tau => zeros(n),  # Creazione di un array di zeri
    :phi => [0.5, 0.1],
    :sigma => 0.5
)

# Punto e valore fissati (pinned point/value)
pinned_point = div(n_time, 2)  # punto fissato (metà del tempo)
pinned_value = mean(dat[:y][pinned_point, :]) # valore medio della colonna `pinned_point` (in R 'apply(dat$y, 1, mean)[pinned_point]')

# Iterazioni di MCMC
n_iter = 3000