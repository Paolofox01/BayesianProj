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
    
    #chain_y[1] = sample_y(...)   #to be defined
    #chain[1][sigma_c][...] = sample_sigma_c
    
    start = time()
    # Iterazioni
    for iter in 2:n_iter
        if (iter/n_iter*100) % 10 == 0.0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        #println(iter)

        current = deepcopy(chain[iter - 1])
        
        for k in 1:K
            current[k][:f] = sample_f_k(k, tt, current)
            current[k][:tau] = sample_tau_k(k, tt, current, hyperparam)
            #println(current[:phi])
            current[k][:beta] = sample_beta_k(k, current, dat[:X])
            current[k][:gamma] = sample_gamma_k(k, tt, current, dat[:X])
            current[k][:phi], current[k][:Sigma_gamma] = sample_phi_k(k, dist, dat[:X], current, hyperparam)
            current[k][:rho], current[k][:Sigma_f], current[k][:Sigma_f_inv] = sample_rho_k(k, tt, current, hyperparam)
            current[k][:h] = sample_h_k(k, y_ict, sigma_c, hyperparam)
            for i in 1:N
                current[k][:g][i,:] = sample_g_ik(i, k, tt, y_ict, current, sigma_c)
            end
        end

        #chain_y[iter] = sample_y(current, chain_sigma2, ..)

        for c in 1:C
            chain_sigma2[iter] = sample_sigma_c(c, current, y_ict, hyperparam)
        end
        
        
        # Registrazione dei campioni dell'iterazione corrente
        chain[iter] = copy(current)
        #chain_y[iter] = get_y(....) # to be defined
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