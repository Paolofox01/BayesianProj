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
function fit_model(tt, K, dat, theta0, n_iter, hyperparam)
    # Inizializzazione delle catene
    chain = Vector{Any}(undef, n_iter)
    chain_y = Vector{Any}(undef, n_iter)
    chain_sigma2 = Vector{Any}(undef, n_iter)

    N = size(dat[:y], 1) #32
    C = size(dat[:y], 2) #6
    T = size(dat[:y], 3) #365

    dist = euclid_dist(dat[:coords][:, 1], dat[:coords][:, 2], N)
    

    # Inizializzazione della prima iterazione della catena
    chain[1] = deepcopy(theta0)

    for k in 1:K
        chain[1][k][:Sigma_f] = sq_exp_kernel(tt, chain[1][k][:rho]) #toeplitz matrix 
        chain[1][k][:Sigma_f_inv] = trench(chain[1][k][:Sigma_f])    #toeplitz matrix fast inversion
        chain[1][k][:f] = sample_f(tt, chain[1][k])
        chain[1][k][:g] = get_mu_g_matrix(tt, chain[1][k])
        chain[1][k][:Sigma_gamma] = get_Sigma_gamma(dist, chain[1][k][:phi])
    end 

    
    start = time()
    # Iterazioni
    for iter in 2:n_iter
        if (iter/n_iter*100) % 10 == 0.0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        #println(iter)

        current = deepcopy(chain[iter - 1])
        
        current[k][:f] = sample_f(tt, current)
        current[k][:tau] = sample_tau(t, g, f, current, hyperparam, Sigma_f, Sigma_f_inv)
        
        #println(current[:phi])
        current[k][:beta] = sample_beta(current, Sigma_gamma, X)
        current[k][:gamma] = sample_gamma(t, g, f, current,  Sigma_f, Sigma_f_inv, Sigma_gamma, X)
        current[k][:phi], current[k][:Sigma_gamma] = sample_phi(dist, X, current, hyperparam)
        current[k][:rho], current[k][:Sigma_f], current[k][:Sigma_f_inv] = sample_rho(t, g, f, current, hyperparam)
       


        #println(isposdef(Sigma_f))
        # Aggiornamento di y_hat
        g_hat = get_mu_g_matrix(g, f, t, current, Sigma_f_inv)
        
        # Calcolo dei residui e campionamento dei parametri dei residui
        z = g - g_hat
        
        # Registrazione dei campioni dell'iterazione corrente
        chain[iter] = copy(current)
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