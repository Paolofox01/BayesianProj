using Distributions

# tau proposal.
#' @param tau Current value of tau.
#' @param tau_proposal_sd Standard deviation of proposal distribution.
function propose_tau(tau, tau_proposal_sd)
    n = size(tau, 1)
    Sigma = tau_proposal_sd^2 * I(n)  # matrice identità di dimensione n
    proposal = rand(MvNormal(tau, Sigma))  # Genera il campione dalla distribuzione normale multivariata
    return proposal
end


# rho proposal.
#' @param log_rho log() ofCurrent value of rho.
#' @param rho_proposal_sd Standard deviation of rho proposal distribution.
function propose_log_rho(log_rho, rho_proposal_sd)
    proposal = rand(Normal(log_rho, rho_proposal_sd))  # Genera un campione dalla normale
    return proposal
end


# phi proposal.
#' @param log_phi log() ofCurrent value of phi.
#' @param phi_proposal_sd Standard deviation of phiproposal distribution.
function propose_log_phi(log_phi, phi_proposal_sd)
    proposal = rand(Normal(log_phi, phi_proposal_sd))  # Genera un campione dalla normale
    return proposal
end


# beta_i proposal. (marginal distribution)
#' @param beta_i Current value of beta_i.
#' @param beta_proposal_sd Standard deviation of proposal distribution.
function propose_beta_i(beta_i, beta_proposal_sd)
    proposal = rand(Normal(beta_i, beta_proposal_sd)) # Genera il campione dalla distribuzione normale
    return proposal
end


# gamma_i proposal. (marginal distribution)
#' @param gamma_i Current value of gamma_i.
#' @param gamma_proposal_sd Standard deviation of gamma proposal distribution.
function propose_gamma_i(gamma_i, gamma_proposal_sd)
    proposal = rand(Normal(gamma_i, gamma_proposal_sd)) # Genera il campione dalla distribuzione normale
    return proposal
end



