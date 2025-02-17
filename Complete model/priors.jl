using Distributions, LinearAlgebra

# Initializing the dictionary
prior = Dict()


# prior[:tau] = function prior_tau(tau, hyperparam)
#     N = length(tau)

#     # Creazione della matrice Sigma
#     Sigma = hyperparam[:tau_prior_sd]^2 * (I(N) - ones(N,N) * (1 / (N + 1)))
    
#     # Definizione della distribuzione normale multivariata
#     mvn = MvNormal(zeros(N), Sigma)
    
#     # Calcolo del logaritmo della densità
#     return logpdf(mvn, tau[1:N])
# end



prior[:tau_i] = function prior_tau(tau, hyperparam)

    # Definizione della distribuzione normale univariata
    uvn = Normal(0, hyperparam[:tau_prior_sd])
    
    # Calcolo del logaritmo della densità
    return logpdf(uvn, tau)
end



prior[:rho] = function prior_rho(rho, hyperparam)
    # Definizione della distribuzione Gamma
    gamma_dist = Gamma(hyperparam[:rho_prior_shape], hyperparam[:rho_prior_scale])
    
    # Calcolo del logaritmo della densità della distribuzione Gamma
    return logpdf(gamma_dist, rho)
end



prior[:phi] = function prior_phi(phi, hyperparam)
    # Definizione della distribuzione Gamma
    gamma_dist = Gamma(hyperparam[:phi_prior_shape], hyperparam[:phi_prior_scale])
    
    # Calcolo del logaritmo della densità della distribuzione Gamma
    return logpdf(gamma_dist, phi)
end



prior[:gamma] = function prior_gamma(gamma, mu_gamma, Sigma_gamma)
    N = length(gamma)
    
    # Definizione della distribuzione normale multivariata
    mvn = MvNormal(mu_gamma, Sigma_gamma)
    
    # Calcolo del logaritmo della densità
    return logpdf(mvn, gamma[1:N])
end



prior[:beta_i] = function prior_beta_i(beta_i)
     # Definizione della distribuzione normale univariata
    uvn = Normal(0, 1)
    
     # Calcolo del logaritmo della densità
    return logpdf(uvn, beta_i)
end



prior[:h] = function prior_h(h_vec, hyperparam)
    # Definizione della distribuzione Dirichlet
    dir = Dirichlet(hyperparam[:prior_h_alpha0])
    
    # Calcolo del logaritmo della densità
    return logpdf(dir, h_vec)
end



prior[:sigma_c] = function prior_sigma_c(sigma_c, hyperparam)
    # Definizione della distribuzione Inverse Gamma
    invga_dist = InverseGamma(hyperparam[:prior_sc_a], hyperparam[:prior_sc_b])
    
    # Calcolo del logaritmo della densità
    return logpdf(invga_dist, sigma_c)
end

