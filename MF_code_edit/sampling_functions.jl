using Distributions
using LinearAlgebra
using Random

include("utilities.jl")


#' Sample tau.
#'
#' @param y Matrix of observed trial data (n_time x n).
#' @param f Vector of f values (n_time).
#' @param current Named list of current parameter values.
#' @param hyperparam Named list of hyperparameter values.
#' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
function sample_tau(t, g, f, current, hyperparam, Sigma_f, Sigma_f_inv)
    # Proposta del nuovo valore di tau
    proposed = copy(current)
    proposed[:tau]= propose_tau(current[:tau], hyperparam[:tau_proposal_sd])
    
    # Calcolo della likelihood e prior per il valore corrente di tau
    lik_current = likelihood(t, g, f, current, Sigma_f, Sigma_f_inv)
    prior_current = prior[:tau](current[:tau], hyperparam[:tau_prior_sd])
    
    # Calcolo della likelihood e prior per il valore proposto di tau
    lik_proposed = likelihood(t, g, f, proposed, Sigma_f, Sigma_f_inv)
    prior_proposed = prior[:tau](proposed[:tau], hyperparam[:tau_prior_sd])
    
    # Calcolo della probabilità di accettazione
    prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
    
    # Decisione sulla proposta in base alla probabilità
    if prob > rand()
        return proposed[:tau]
    else
        return current[:tau]
    end
end



# Sample rho.
#' @param g Matrix of observed trial data (n_time x n).
#' @param f Vector of f values (n_time).
#' @param current Named list of current parameter values.
#' @param hyperparam Named list of hyperparameter values.
function sample_rho(t, g, f, current, hyperparam)
    n_time = size(g, 2)
    
    proposed = copy(current)
    proposed[:rho] = exp( propose_log_rho(log(current[:rho]), hyperparam[:rho_proposal_sd]) )
    Sigma_f_prop = sq_exp_kernel(t, proposed[:rho], nugget = 1e-6)
    Sigma_f_prop_inv = inv(Sigma_f_prop)
    
    Sigma_f_curr = sq_exp_kernel(t, current[:rho], nugget = 1e-6)
    Sigma_f_curr_inv = inv(Sigma_f_curr)
    
    lik_current = likelihood(t, g, f, current, Sigma_f_curr, Sigma_f_curr_inv)
    prior_current = prior[:rho](current[:rho], hyperparam[:rho_prior_shape], hyperparam[:rho_prior_scale])
    
    lik_proposed = likelihood(t, g, f, proposed, Sigma_f_prop, Sigma_f_prop_inv)
    prior_proposed = prior[:rho](proposed[:rho], hyperparam[:rho_prior_shape], hyperparam[:rho_prior_scale])
    
    prob = exp(lik_proposed + prior_proposed + proposed[:rho] - lik_current - prior_current - current[:rho])
    
    if prob > rand()
        return proposed[:rho]
    else
        return current[:rho]
    end
end






# Sample from posterior of f.
#' @param y Matrix of observed trial data (n_time x n).
#' @param theta Named list of parameter values.
#' @param n_draws Numbver of draws.
#' @param nugget GP covariance nugget.
function sample_f(g, theta, n_draws; nugget = 1e-6)
    n_time = size(g, 2)
    n = size(g, 1)
    chain_f = Vector{Any}(undef, n_draws)
    t = range(1, stop=n_time, length=n_time)
    Sigma_f = sq_exp_kernel(t, theta[:rho], nugget = nugget)
    Sigma_f_inv = inv(Sigma_f)
    
    for iter in 1:n_draws
        if iter % 10 == 0
            println(iter / 10)
        end
        A = Sigma_f_inv
        b = zeros(n_time)
        
        for i in 1:n
            Sigma_g_i = get_Sigma_g_i(i, t, theta, Sigma_f, Sigma_f_inv)
            Sigma_i = get_Sigma_i(i, t, theta)
            #Sigma_i = (Sigma_i + Sigma_i') / 2
            
            Sigma_g_i_inv = inv(Sigma_g_i)
            L = Sigma_i * Sigma_f_inv
            G = Sigma_g_i_inv * L
            A += L' * G
            b += (g[i, :]' * G)[:]
        end

        Sigma_f_post = inv(A)
        Sigma_f_post = (Sigma_f_post + Sigma_f_post') / 2
        chain_f[iter] = rand(MultivariateNormal(Sigma_f_post * b, Sigma_f_post))
    end
    
    f_draws = hcat(chain_f...)
    return f_draws
end







function sample_gamma(t, g, f, current,  Sigma_f, Sigma_f_inv, Sigma_gamma, X)
    #n_time = size(g, 2)
    n = size(g, 1)
    curr = copy(current)

    for i in 1:n 
        proposed = copy(curr)
        proposed[:gamma][i] = propose_gamma_i(curr[:gamma][i], hyperparam[:gamma_proposal_sd])
        #println("gamma_i: ", proposed[:gamma][i])

        lik_current = target_g_i(i, t, g, f, curr, Sigma_f, Sigma_f_inv)
        prior_current = marginal_gamma_i(i, curr, Sigma_gamma, X)
        
        lik_proposed = target_g_i(i, t, g, f, proposed, Sigma_f, Sigma_f_inv)
        prior_proposed = marginal_gamma_i(i, proposed, Sigma_gamma, X)
        
        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)

        if prob > rand()
            curr[:gamma][i] = copy(proposed[:gamma][i])
        end
    end
    return curr[:gamma]
end





# # Sample betas.
# #' @param y Matrix of observed trial data (n_time x n).
# #' @param f Vector of f values (n_time).
# #' @param current Named list of current parameter values.
# #' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
# #' @param y_hat
# #' @param hyperparam Named list of hyperparameter values.
function sample_beta(current, hyperparam, Sigma_gamma, X)
    curr = copy(current)

    for i in 1:size(X, 2)
        proposed = copy(curr)
        proposed[:beta][i]= propose_beta_i(curr[:beta][i], hyperparam[:beta_proposal_sd])

        lik_current = target_gamma(curr, Sigma_gamma, X)
        prior_current = prior[:beta_i](curr[:beta][i])
    
        # Calcolo della likelihood e prior per il valore proposto di tau
        lik_proposed = target_gamma(proposed, Sigma_gamma, X)
        prior_proposed = prior[:beta_i](proposed[:beta][i])

        #println(lik_current, "   ",  prior_current, "   ", lik_proposed, "   ", prior_proposed)
        # Calcolo della probabilità di accettazione
        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)

        if prob > rand()
            curr[:beta][i] = copy(proposed[:beta][i])
        end
    end

    return curr[:beta]
    
end




function sample_phi(D, X, current, hyperparam)
    
    proposed = copy(current)
    proposed[:phi] = exp( propose_log_phi(log(current[:phi]), hyperparam[:phi_proposal_sd]) )
    
    Sigma_gamma_prop = get_Sigma_gamma(D, proposed[:phi])
    Sigma_gamma_curr = get_Sigma_gamma(D, current[:phi])
    
    lik_current = target_gamma(current, Sigma_gamma_curr, X)
    prior_current = prior[:phi](current[:phi], hyperparam[:phi_prior_shape], hyperparam[:phi_prior_scale])
    
    lik_proposed = target_gamma(proposed, Sigma_gamma_prop, X)
    prior_proposed = prior[:phi](proposed[:phi], hyperparam[:phi_prior_shape], hyperparam[:phi_prior_scale])
    
    prob = exp(lik_proposed + prior_proposed + proposed[:phi] - lik_current - prior_current - current[:phi])
    
    if prob > rand()
        return proposed[:phi], Sigma_gamma_prop
    else
        return current[:phi], Sigma_gamma_curr
    end
end
