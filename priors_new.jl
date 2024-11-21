using Distributions, LinearAlgebra

#' Prior for tau (latency).
#'
#' @param tau tau value.
#' @param tau_prior_sd SD of prior distribution.
prior = Dict()

# Funzione per prior_tau
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

#' Prior for rho (GP length scale).
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

prior[:rho_w] = function prior_rho_w(rho_w, rho_w_prior_shape, rho_w_prior_scale)
    gamma_dist = InverseGamma(rho_w_prior_shape, rho_w_prior_scale)
    
    # Calcolo del logaritmo della densità della distribuzione Gamma
    result = logpdf(gamma_dist, rho_w)
    
    return result
end


prior[:beta_k] = function prior_beta_k(beta_k)
    
    result = 0.0
    n = length(beta_k)

    
    # Definizione della distribuzione normale multivariata
    mvn = MvNormal(zeros(n), I(n))
    
    # Calcolo del logaritmo della densità
    result += logpdf(mvn, beta_k[1:n])
    
    return result

end