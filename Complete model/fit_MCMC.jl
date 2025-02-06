using LinearAlgebra

include("utilities.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")

#' Fit BSA Model
#'
#' @param y data
#' @param n_iter number of posterior samples
#' @param theta0 parameter initializations
#' @param hyperparam hyperparameters
#' @param pinned_point pinned point for identifiability of structural signal
#' @param pinned_value value of f at the pinned point


# Funzione per il fitting del modello RPAGP (Random Process Approximate GP)
function fit_model(sites, g, n_iter, theta0, hyperparam)
    # Inizializzazione delle catene
    chain = Vector{Any}(undef, n_iter)
    chain_f = Vector{Any}(undef, n_iter)
    chain_g = Vector{Any}(undef, n_iter)
    chain_z = Vector{Any}(undef, n_iter)
    n_time = size(g, 2) #365
    n = size(g, 1) #32

    t = range(1, stop=n_time, length=n_time)
    dist = euclid_dist(sites[:, 1], sites[:, 2], n)
    X = sites[:, 3:6]
    

    # Inizializzazione della prima iterazione della catena
    chain[1] = copy(theta0)

    chain_f[1] = sample_f(g, chain[1], 1)
    Sigma_f = sq_exp_kernel(t, chain[1][:rho], nugget = 1e-9)
    Sigma_f_inv = inv(Sigma_f)
    println(isposdef(Sigma_f))

    chain_g[1] = get_mu_g_matrix(g, chain_f[1], t, chain[1], Sigma_f_inv)

    chain_z[1] = g - chain_g[1]
    
    start = time()

    Sigma_gamma = get_Sigma_gamma(dist, chain[1][:phi])

    # Iterazioni
    for iter in 2:n_iter
        if (iter/n_iter*100) % 10 == 0.0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        #println(iter)

        curr = copy(chain[iter - 1])
        
        
        # Campionamento di tau e rho
        f = sample_f(g, curr, 1)
        
        curr[:beta] = sample_beta(curr, hyperparam, Sigma_gamma, X)
        #println("beta: ", curr[:beta])

        curr[:phi], Sigma_gamma = sample_phi(dist, X, curr, hyperparam)
        #println("phi: ", curr[:phi])
        #isposdef(Sigma_gamma)

        curr[:gamma] = sample_gamma(t, g, f, curr,  Sigma_f, Sigma_f_inv, Sigma_gamma, X)
        #println("gamma: ", curr[:gamma])

        curr[:tau] = sample_tau(t, g, f, curr, hyperparam, Sigma_f, Sigma_f_inv)
        #println("tau: ", curr[:tau])

        curr[:rho] = sample_rho(t, g, f, curr, hyperparam)
        #println("rho: ", curr[:rho])

        Sigma_f = sq_exp_kernel(t, curr[:rho], nugget = 1e-9)
        Sigma_f_inv = inv(Sigma_f)
        #println(isposdef(Sigma_f))
        # Aggiornamento di y_hat
        g_hat = get_mu_g_matrix(g, f, t, curr, Sigma_f_inv)
        
        # Calcolo dei residui e campionamento dei parametri dei residui
        z = g - g_hat
        
        # Registrazione dei campioni dell'iterazione corrente
        chain_f[iter] = copy(f)
        chain[iter] = copy(curr)
        chain_g[iter] = copy(g_hat)
        chain_z[iter] = copy(z)
        
        #println(iter, "  ", chain[iter])
        # println(iter, "   ", curr[:rho])
    end

    
    println("\n")
    fine = time()
    # Stampa del tempo di esecuzione
    runtime = fine - start
    println("Tempo di esecuzione: ", runtime)

    # Restituzione del risultato come dizionario
    return Dict(
        :chain => chain,
        :chain_f => chain_f,
        :chain_g => chain_g,
        :chain_z => chain_z,
        :runtime => runtime
    )
end