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
    n_time = size(g, 1)
    x = LinRange(0.0, 1.0, n_time)
    
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


# Sample betas.
#' @param y Matrix of observed trial data (n_time x n).
#' @param f Vector of f values (n_time).
#' @param current Named list of current parameter values.
#' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
#' @param y_hat
#' @param hyperparam Named list of hyperparameter values.
function sample_beta(g, current, g_hat, hyperparam)
    n = size(g, 1)
    betas = []
    
    for i in 1:n
        betas = push!(betas, sample_blr(g[i, :], g_hat[:, i] / current[:beta][i],
                                        hyperparam[:beta_prior_mu], hyperparam[:beta_prior_sd],
                                        Matrix{Float64}(I, 1, 1), 1, 1)[:beta])
    end
    
    return betas
end


# One draw from the posterior of a Bayesian Linear Regression.
#' @param y Response vector.
#' @param X Design matrix.
#' @param mu Prior mean of coefficients.
#' @param sigma Prior standard deviation of coefficients.
#' @param V Prior covariance of coefficients.
#' @param a Hyperparameter for noise variance.
#' @param b Hyperparameter for noise variance.
function sample_blr(g, X, mu, sigma, V, a, b)
    n = length(g)
    V_post = inv(inv(V) .+ X' * X)
    mu_post = V_post * (inv(V) * mu .+ X' * g)
    a_post = a + n / 2
    b_post = b .+ 1 / 2 * (mu' * inv(V) * mu .+ g' * g .- mu_post' * inv(V_post) * mu_post)
    
    beta = rand(MvNormal(mu_post[:,1], (b_post[1,1] / a_post) * V_post))[1]
    sigma2 = [rand(InverseGamma(a_post, k)) for k in b_post[:,1]]
    
    return Dict(:beta => beta, :sigma2 => sigma2)
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
    x = LinRange(1.0, 365.0, n_time)
    K_f = sq_exp_kernel(x, theta[:rho], nugget = nugget)
    K_f_inv = inv(K_f)
    
    for iter in 1:n_draws
        if iter % 10 == 0
            println(iter / 10)
        end
        A = K_f_inv
        b = zeros(n_time)
        
        for i in 1:n
            Sigma_g_i = get_Sigma_g_i((theta[:beta])[i], K_f)
            K_i = get_K_i(x, Dict(:rho => theta[:rho], :tau => theta[:tau][i], :gamma => theta[:beta][i]))
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