using Distributions


# function propose_tau(tau, tau_proposal_sd)
#     n = size(tau, 1)
#     Sigma = tau_proposal_sd^2 * I(n)  # matrice identità di dimensione n
#     proposal = rand(MvNormal(tau, Sigma))  # Genera il campione dalla distribuzione normale multivariata
#     return proposal
# end



function propose_tau_i(tau_i, hyperparam)
    return rand(Normal(tau_i, hyperparam[:tau_proposal_sd]))  # Genera il campione dalla distribuzione normale multivariata 
end



function propose_log_rho(log_rho, hyperparam)
    return rand(Normal(log_rho, hyperparam[:rho_proposal_sd]))  # Genera un campione dalla normale
end



function propose_log_phi(log_phi, hyperparam)
    return rand(Normal(log_phi, hyperparam[:phi_proposal_sd]))  # Genera un campione dalla normale
end



function propose_beta_i(beta_i, hyperparam)
    return rand(Normal(beta_i, hyperparam[:beta_proposal_sd])) # Genera il campione dalla distribuzione normale
end



function propose_gamma_i(gamma_i, hyperparam)
    return rand(Normal(gamma_i, hyperparam[:gamma_proposal_sd])) # Genera il campione dalla distribuzione normale
end



function propose_h_alr(h_vec, hyperparam)
    C = length(h_vec)
    return rand(MultivariateNormal( log.(h_vec[1:(C-1)]./h_vec[C]), hyperparam[:h_proposal_sd].*I(C-1))) # Genera il campione dalla distribuzione normale
end
