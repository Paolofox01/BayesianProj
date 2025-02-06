using Distributions, LinearAlgebra

# Initializing the dictionary
prior = Dict()


#' Prior for tau (latency).
#'
#' @param tau tau value.
#' @param tau_prior_sd SD of prior distribution.
prior[:tau] = function prior_tau(tau, tau_prior_sd)
    result = 0.0
    n = length(tau)
    
    # Creazione della matrice Sigma
    Sigma = tau_prior_sd^2 * (I(n) - ones(n, n) * (1 / (n + 1)))
    
    # Definizione della distribuzione normale multivariata
    mvn = MvNormal(zeros(n), Sigma)
    
    # Calcolo del logaritmo della densità
    result += logpdf(mvn, tau[1:n])
    
    return result
end


#' Prior for rho (temporal GP length scale).
#'
#' @param rho rho value.
#' @param rho_prior_shape Shape parameter of prior.
#' @param rho_prior_scale Scale parameter of prior.
prior[:rho] = function prior_rho(rho, rho_prior_shape, rho_prior_scale)
    gamma_dist = Gamma(rho_prior_shape, rho_prior_scale)
    
    # Calcolo del logaritmo della densità della distribuzione Gamma
    result = logpdf(gamma_dist, rho)
    
    return result
end


#' Prior for phi (spatial GP length scale).
#'
#' @param phi phi value.
#' @param phi_prior_shape Shape parameter of prior.
#' @param phi_prior_scale Scale parameter of prior.
prior[:phi] = function prior_phi(phi, phi_prior_shape, phi_prior_scale)
    gamma_dist = Gamma(phi_prior_shape, phi_prior_scale)
    
    # Calcolo del logaritmo della densità della distribuzione Gamma
    result = logpdf(gamma_dist, phi)
    
    return result
end


#' Prior for gamma.
#'
#' @param gamma gamma value.
#' @param phi_prior_shape Shape parameter of prior.
#' @param phi_prior_scale Scale parameter of prior.
prior[:gamma] = function prior_gamma(gamma, Sigma_gamma, mu_gamma)
    n = length(gamma)
    
    # Definizione della distribuzione normale multivariata
    mvn = MvNormal(mu_gamma, Sigma_gamma)
    
    # Calcolo del logaritmo della densità
    result = logpdf(mvn, gamma[1:n])
    
    return result

end



#' Marginal prior for beta_i.
#'
#' @param beta_i beta_i value.
prior[:beta_i] = function prior_beta_i(beta_i)
     # Definizione della distribuzione normale univariata
    mvn = Normal(0, 1)
    
     # Calcolo del logaritmo della densità
    result = logpdf(mvn, beta_i)
    
    return result
end

