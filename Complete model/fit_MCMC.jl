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
function fit_model(sites, g, n_iter, theta0, hyperparam, f0_check)
    # Inizializzazione delle catene
    chain = Vector{Any}(undef, n_iter)
    chain_f = Vector{Any}(undef, n_iter)
    chain_g = Vector{Any}(undef, n_iter)
    chain_z = Vector{Any}(undef, n_iter)
    chain_y = Vector{Any}(undef, n_iter)
    n_time = size(g, 2) #365
    n = size(g, 1) #32

    t = range(1, stop=n_time, length=n_time)
    dist = euclid_dist(sites[:, 1], sites[:, 2], n)
    X = sites[:, 3:6]
    

    # Inizializzazione della prima iterazione della catena
    chain[1] = deepcopy(theta0)
    Sigma_f = sq_exp_kernel(t, chain[1][:rho]) #toeplitz matrix 
    #Sigma_f_inv = inv(Sigma_f)
    Sigma_f_inv = trench(Sigma_f) #toeplitz matrix fast inversion

    chain_f[1] = sample_f(g, t, chain[1], Sigma_f, Sigma_f_inv)
    #chain_f[1] = f0_check

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

        current = deepcopy(chain[iter - 1])
        
        f = sample_f(g, t, current)
        #f = f0_check

        current[:tau] = sample_tau(t, g, f, current, hyperparam, Sigma_f, Sigma_f_inv)
        

        #println(current[:phi])
        current[:beta] = sample_beta(current, Sigma_gamma, X)
        
        current[:gamma] = sample_gamma(t, g, f, current,  Sigma_f, Sigma_f_inv, Sigma_gamma, X)
        
        current[:phi], Sigma_gamma = sample_phi(dist, X, current, hyperparam)

        current[:rho], Sigma_f, Sigma_f_inv = sample_rho(t, g, f, current, hyperparam)
       

        #Sigma_f = sq_exp_kernel(t, current[:rho])
        #Sigma_f_inv = inv(Sigma_f)
        #Sigma_f_inv = trench(Sigma_f)

        #println(isposdef(Sigma_f))
        # Aggiornamento di y_hat
        g_hat = get_mu_g_matrix(g, f, t, current, Sigma_f_inv)
        
        # Calcolo dei residui e campionamento dei parametri dei residui
        z = g - g_hat
        
        # Registrazione dei campioni dell'iterazione corrente
        chain_f[iter] = deepcopy(f)
        chain[iter] = deepcopy(current)
        chain_g[iter] = deepcopy(g_hat)
        chain_z[iter] = deepcopy(z)
        
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