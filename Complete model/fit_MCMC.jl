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
    chain_sigma2_c = Vector{Any}(undef, n_iter)

    N = size(dat[:y], 1) #32
    C = size(dat[:y], 2) #6
    T = size(dat[:y], 3) #365

    dist = euclid_dist(dat[:coords][:, 1], dat[:coords][:, 2], N)
    

    # Inizializzazione della prima iterazione della catena
    chain[1] = deepcopy(theta0)
    chain_sigma2_c[1] = zeros(C)
    for c in 1:C
        chain_sigma2_c[1][c] = var(dat[:y][:,c,:])
    end

    for k in 1:K
        chain[1][k][:Sigma_f] = sq_exp_kernel(tt, chain[1][k][:rho]) #toeplitz matrix 
        chain[1][k][:Sigma_f_inv] = trench(chain[1][k][:Sigma_f])    #toeplitz matrix fast inversion
        chain[1][k][:f] = sample_f_k(tt, chain[1][k])
        chain[1][k][:g] = get_mu_g_matrix(tt, chain[1][k])
        chain[1][k][:Sigma_gamma] = get_Sigma_gamma(dist, chain[1][k][:phi])
    end 
    
    chain_y[1] = sample_y_post(chain[1], chain_sigma2_c[1], N, C, T)
    
    start = time()
    # Iterazioni
    for iter in 2:n_iter
        if (iter/n_iter*100) % 10 == 0.0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        #println(iter)

        current = deepcopy(chain[iter - 1])
        chain_sigma2_c[iter] = zeros(C)
        for c in 1:C
            chain_sigma2_c[iter][c] = sample_sigma2_c(c, current, dat[:y], hyperparam)
        end

        for k in 1:K
            current[k][:f] = sample_f_k(tt, current[k])
            current[k][:tau] = sample_tau_k(tt, current[k], hyperparam)
            #println(current[:phi])
            current[k][:beta] = sample_beta_k(current[k], dat[:X])
            current[k][:gamma] = sample_gamma_k(tt, current[k], dat[:X])
            current[k][:phi], current[k][:Sigma_gamma] = sample_phi_k(dist, dat[:X], current[k], hyperparam)
            current[k][:rho], current[k][:Sigma_f], current[k][:Sigma_f_inv] = sample_rho_k(tt, current[k], hyperparam)
            current[k][:h] = sample_h_k(k, dat[:y], chain_sigma2_c[iter], current, hyperparam)
            for i in 1:N
                current[k][:g][i,:] = sample_g_ik(i, k, tt, dat[:y], current, chain_sigma2_c[iter])
            end
        end
        chain[iter] = copy(current)
        
        
        # Registrazione dei campioni dell'iterazione corrente
        chain_y[iter] = sample_y_post(current, chain_sigma2_c[iter], N, C, T)
    end

    
    println("\n")
    fine = time()
    # Stampa del tempo di esecuzione
    runtime = fine - start
    println("Tempo di esecuzione: ", runtime)

    # Restituzione del risultato come dizionario
    return Dict(
        :chain => chain,
        :chain_y => chain_y,
        :chain_sigma2_c => chain_sigma2_c,
        :runtime => runtime
    )
end