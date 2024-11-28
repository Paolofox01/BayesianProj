using LinearAlgebra

#' Fit RPAGP Model
#'
#' @param y data
#' @param n_iter number of posterior samples
#' @param theta0 parameter initializations
#' @param hyperparam hyperparameters
#' @param pinned_point pinned point for identifiability of structural signal
#' @param pinned_value value of f at the pinned point

# Funzione per il fitting del modello RPAGP (Random Process Approximate GP)
function fit_rpagp(sites, g, n_iter, theta0, hyperparam, pinned_point, pinned_value = 1)
    # Inizializzazione delle catene
    chain = Vector{Any}(undef, n_iter)
    chain_f = Vector{Any}(undef, n_iter)
    chain_g_hat = Vector{Any}(undef, n_iter)
    chain_z = Vector{Any}(undef, n_iter)
    n_time = size(g, 2)
    n = size(g, 1)
    
    x = range(1, stop=n_time, length=n_time)
    
    # Inizializzazione della prima iterazione della catena
    chain[1] = copy(theta0)

  
    chain_f[1] = sample_f(g, chain[1], 1)
    K_f = sq_exp_kernel(x, chain[1][:rho], nugget = 1e-9)
    K_f_inv = inv(K_f)
    
    start = time()

    

    dist = euclid_dist(sites[:, 1], sites[:, 2], n)

    rho_spatial = 400 # per ora lo metto io a mano

    K_spat = 0.5 * exp.(-1 ./ rho_spatial .* dist)

    M = 1. * I(4)

    M[4,4] = 0.01
    
    K_gamma = sites[:, 3:6] * M * sites[:, 3:6]' + K_spat

    K_gamma = (K_gamma' + K_gamma)/2

    # Iterazioni
    for iter in 2:n_iter
        if iter % (n_iter ÷ 10) == 0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        
        current = copy(chain[iter - 1])
        
        # Campionamento di f e ridimensionamento
        #f = sample_f(g, current, 1)
        #f .= pinned_value * f ./ f[pinned_point]
        
        # Aggiornamento di y_hat
        #g_hat = get_g_hat_matrix(g, f, current, K_f_inv)
        
        # Campionamento dei betas
        #  current[:beta] = sample_beta(g, current, g_hat, hyperparam)
        
        # Campionamento di tau e rho
        f = sample_f(g, current, 1)
        # f .= pinned_value * f ./ f[pinned_point]

        current[:beta] = sample_gamma(g, f, current, hyperparam, K_f, K_f_inv, K_gamma)
        current[:tau] = sample_tau(g, f, current, hyperparam, K_f, K_f_inv)
        current[:rho] = sample_rho(g, f, current, hyperparam)
        K_f = sq_exp_kernel(x, current[:rho], nugget = 1e-6)
        K_f_inv = inv(K_f)
        
        # Aggiornamento di y_hat
        g_hat = get_g_hat_matrix(g, f, current, K_f_inv)
        
        # Calcolo dei residui e campionamento dei parametri dei residui
        z = g - g_hat
        
        # Registrazione dei campioni dell'iterazione corrente
        chain_f[iter] = copy(f)
        chain[iter] = copy(current)
        chain_g_hat[iter] = copy(g_hat)
        chain_z[iter] = copy(z)

        # println(iter, "   ", current[:rho])
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
        :chain_g_hat => chain_g_hat,
        :chain_z => chain_z,
        :runtime => runtime
    )
end