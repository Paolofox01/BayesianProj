using Distributions
using LinearAlgebra
using Random

include("utilities_new.jl")


#' Sample tau.
#'
#' @param y Matrix of observed trial data (n_time x n).
#' @param f Vector of f values (n_time).
#' @param current Named list of current parameter values.
#' @param hyperparam Named list of hyperparameter values.
#' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
function sample_tau(g, f, current, hyperparam, K_f, K_f_inv)
    # Proposta del nuovo valore di tau
    proposed = copy(current)
    proposed[:tau]= propose_tau(current[:tau], hyperparam[:tau_proposal_sd])
    
    # Calcolo della likelihood e prior per il valore corrente di tau
    lik_current = likelihood(g, f, current, K_f, K_f_inv)
    prior_current = prior[:tau](current[:tau], hyperparam[:tau_prior_sd])
    
    # Calcolo della likelihood e prior per il valore proposto di tau
    lik_proposed = likelihood(g, f, proposed, K_f, K_f_inv)
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
#' @param y Matrix of observed trial data (n_time x n).
#' @param f Vector of f values (n_time).
#' @param current Named list of current parameter values.
#' @param hyperparam Named list of hyperparameter values.
#' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
function sample_rho(g, f, current, hyperparam)
    n_time = size(g, 2)
    x = range(1, stop=n_time, length=n_time)
    
    proposed = copy(current)
    proposed[:rho] = propose_rho(current[:rho], hyperparam[:rho_proposal_sd])
    
    K_f_curr = sq_exp_kernel(x, current[:rho], nugget = 1e-6)
    K_f_curr_inv = inv(K_f_curr)
    
    K_f_prop = sq_exp_kernel(x, proposed[:rho], nugget = 1e-6)
    K_f_prop_inv = inv(K_f_prop)
    
    lik_current = likelihood(g, f, current, K_f_curr, K_f_curr_inv)
    prior_current = prior[:rho](current[:rho], hyperparam[:rho_prior_shape], hyperparam[:rho_prior_scale])
    
    lik_proposed = likelihood(g, f, proposed, K_f_prop, K_f_prop_inv)
    prior_proposed = prior[:rho](proposed[:rho], hyperparam[:rho_prior_shape], hyperparam[:rho_prior_scale])
    
    prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
    
    if prob > rand()
        return proposed[:rho]
    else
        return current[:rho]
    end
end


# # Sample betas.
# #' @param y Matrix of observed trial data (n_time x n).
# #' @param f Vector of f values (n_time).
# #' @param current Named list of current parameter values.
# #' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
# #' @param y_hat
# #' @param hyperparam Named list of hyperparameter values.
# function sample_beta(g, current, g_hat, hyperparam)
#     n = size(g, 1)
#     betas = zeros(n)
    
#     for i in 1:n
#         betas[i] += sample_blr(g[i, :], g_hat[i, :] / current[:beta][i],
#                                         hyperparam[:beta_prior_mu], hyperparam[:beta_prior_sd],
#                                         Matrix{Float64}(I, 1, 1), 1, 1)[:beta][1]
#     end
    
#     return betas
# end


# # One draw from the posterior of a Bayesian Linear Regression.
# #' @param y Response vector.
# #' @param X Design matrix.
# #' @param mu Prior mean of coefficients.
# #' @param sigma Prior standard deviation of coefficients.
# #' @param V Prior covariance of coefficients.
# #' @param a Hyperparameter for noise variance.
# #' @param b Hyperparameter for noise variance.

# # Ho fatto un po di magheggi con sta funzione perchè per qualche motivo V è una matrice con un uno e non è automatica la conversione matrice reale
# # alla fine sono tutti reali e avrei potuto lasciare solo quelli ma volevo mantenere la generalità per la MV T-student
# # che poi alla fine se fa l'inverse gamma è perchè sa che b_post è uno sclare però bo non l'ho capita la Pluta qui

# function sample_blr(g, X, mu, sigma, V, a, b)
#     n = size(g, 1) # 365 in questo caso, è solo una riga
    
#     V_post = inv(inv(V) .+ X' * X) # attenzione, qui sia X' * X che V hanno dimensione 1x1 ma per qualche motivo V è passata come matrice, perciò metto .+ per far fare l'addizione, altrimenti va modificato il tipo di V 
#     mu_post = V_post * (inv(V) * mu .+ X' * g) #uguale a sopra
#     a_post = a + n / 2
#     b_post = b .+ 1 / 2 * (mu' * inv(V) * mu .+ g' * g .- mu_post' * inv(V_post) * mu_post)
    
#     beta = rand(MvTDist(2 * a_post, mu_post[:, 1], (b_post / a_post) * V_post))


#     sigma2 = rand(InverseGamma(a_post, b_post[1,1]))
    
#     return Dict(:beta => beta, :sigma2 => sigma2)
# end


# Sample from posterior of f.
#' @param y Matrix of observed trial data (n_time x n).
#' @param theta Named list of parameter values.
#' @param n_draws Numbver of draws.
#' @param nugget GP covariance nugget.
function sample_f(g, theta, n_draws; nugget = 1e-6)
    n_time = size(g, 2)
    n = size(g, 1)
    chain_f = Vector{Any}(undef, n_draws)
    x = range(1, stop=n_time, length=n_time)
    K_f = sq_exp_kernel(x, theta[:rho], nugget = nugget)
    K_f_inv = inv(K_f)
    
    for iter in 1:n_draws
        if iter % 10 == 0
            println(iter / 10)
        end
        A = K_f_inv
        b = zeros(n_time)
        
        for i in 1:n
            Sigma_g_i = get_Sigma_g_i(theta[:gamma][i], K_f)
            K_i = get_K_i(x, Dict(:rho => theta[:rho], :tau => theta[:tau][i], :gamma => theta[:gamma][i]))
            Sigma_i = Sigma_g_i - K_i' * K_f_inv * K_i
            Sigma_i = (Sigma_i + Sigma_i') / 2
            
            Sigma_i_inv = inv(Sigma_i)
            L = K_i * K_f_inv
            G = Sigma_i_inv * L
            A += L' * G
            b += (g[i, :]' * G)[:]
        end

        K_f_post = inv(A)
        K_f_post = (K_f_post + K_f_post') / 2
        chain_f[iter] = rand(MultivariateNormal(K_f_post * b, K_f_post))
    end
    
    f_draws = hcat(chain_f...)
    return f_draws
end


function sample_gamma(g, f, current, K_f_inv, K_spat, sites; nugget = 0.0)

    n_time = size(g, 2)
    n = size(g, 1)
    x = range(1, stop=n_time, length=n_time)

    mu_loggamma = sites * current[:beta]

    mu_gamma = exp.(mu_loggamma + diag(K_spat) ./ 2)

    K_gamma = Diagonal(mu_gamma) * (exp.(K_spat) .- 1) * Diagonal(mu_gamma)

    K_loggamma_g, f_matrix, mu_g = get_sigma_loggamma_g(n, n_time, f, K_spat, mu_loggamma, mu_gamma, current, K_f_inv)

    K_g = f_matrix * K_gamma * f_matrix' + nugget * I(n*n_time) 

    K_g_inv = inv(K_g)

    mu_loggamma_dato_g = mu_loggamma + K_loggamma_g * K_g_inv * (vec(g) - mu_g) #le g vanno messe in colonna

    cov_loggamma_dato_g = K_spat - K_loggamma_g * K_g_inv * K_loggamma_g'

    cov_loggamma_dato_g = (cov_loggamma_dato_g + cov_loggamma_dato_g')/2

    println(det(cov_loggamma_dato_g))

    println(size(cov_loggamma_dato_g))

    loggamma = rand(MvNormal(mu_loggamma_dato_g, cov_loggamma_dato_g))

    return exp.(loggamma)


end

# function sample_gamma(g, f, current, hyperparam, K_f, K_f_inv, K_spat, sites)
#     # Proposta del nuovo valore di tau
#     proposed = copy(current)
#     proposed[:gamma]= propose_gamma(current[:gamma], hyperparam[:gamma_proposal_sd])
    
#     # Calcolo della likelihood e prior per il valore corrente di gamma
#     lik_current = likelihood(g, f, current, K_f, K_f_inv)
#     prior_current = prior[:gamma](current[:gamma], K_spat, sites * current[:beta])
    
#     # Calcolo della likelihood e prior per il valore proposto di tau
#     lik_proposed = likelihood(g, f, proposed, K_f, K_f_inv)
#     prior_proposed = prior[:gamma](proposed[:gamma], K_spat, sites * current[:beta])
    
#     # Calcolo della probabilità di accettazione
#     prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
    
#     # Decisione sulla proposta in base alla probabilità
#     if prob > rand()
#         return proposed[:gamma]
#     else
#         return current[:gamma]
#     end
# end


function sample_beta(current, hyperparam, K_spat, sites)
    
    for i in 1:size(sites, 2)
        proposed = copy(current)
        proposed[:beta][i]= propose_beta_i(current[:beta][i], hyperparam[:beta_proposal_sd])

        lik_current = likelihood_gamma(current[:gamma], current[:beta], K_spat, sites)
        prior_current = prior[:beta_i](current[:beta][i])
    
        # Calcolo della likelihood e prior per il valore proposto di tau
        lik_proposed = likelihood_gamma(current[:gamma], proposed[:beta], K_spat, sites)
        prior_proposed = prior[:beta_i](proposed[:beta][i])

        #println(lik_current, "   ",  prior_current, "   ", lik_proposed, "   ", prior_proposed)
        # Calcolo della probabilità di accettazione
        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)

        if prob > rand()
            current[:beta] = copy(proposed[:beta])
        end

    end

    #println(lik_proposed + prior_proposed - lik_current - prior_current)

    #println(prob)

    return current[:beta]
    
end


function sample_rho_spatial(current, hyperparam, K_spat, sites, dist)
    proposed = copy(current)
    proposed[:rho_spatial]= propose_rho_spatial(current[:rho_spatial], hyperparam[:rho_spatial_proposal_sd])

    lik_current = likelihood_gamma(current[:gamma], current[:beta], K_spat, sites)                              
    prior_current = prior[:rho_spatial](current[:rho_spatial], hyperparam[:rho_spatial_prior_shape], hyperparam[:rho_spatial_prior_scale])
    
    K_spat_proposed = exp.(-1 ./ proposed[:rho_spatial] .* dist)

    # Calcolo della likelihood e prior per il valore proposto di tau
    lik_proposed = likelihood_gamma(current[:gamma], current[:beta], K_spat_proposed, sites)
    prior_proposed = prior[:rho_spatial](proposed[:rho_spatial], hyperparam[:rho_spatial_prior_shape], hyperparam[:rho_spatial_prior_scale])

    # Calcolo della probabilità di accettazione
    prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)

    # println(lik_proposed ," + ", prior_proposed, " - ",  lik_current, " - ",  prior_current, " = ", prob)
    
    # Decisione sulla proposta in base alla probabilità
    if prob > rand()
        return proposed[:rho_spatial], K_spat_proposed
    else
        return current[:rho_spatial], K_spat
    end
end