using Distributions

# tau proposal.
#' @param tau Current value of tau.
#' @param tau_proposal_sd Standard deviation of proposal distribution.
function propose_tau(tau, tau_proposal_sd)
    n = length(tau)
    Sigma = tau_proposal_sd^2 * I(n)  # matrice identità di dimensione n
    proposal = rand(MvNormal(tau, Sigma))  # Genera il campione dalla distribuzione normale multivariata
    return proposal
end


# rho proposal.
#' @param rho Current value of rho.
#' @param rho_proposal_sd Standard deviation of rho proposal distribution.
function propose_rho(rho, rho_proposal_sd)
    proposal = rand(Normal(rho, rho_proposal_sd))  # Genera un campione dalla normale
    return proposal
end