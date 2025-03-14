using Distributions, LinearAlgebra

include("utilities.jl")

function target_g_k_prod(tt, theta)
    N = size(theta[:g], 1)
    T = size(theta[:g], 2)
    
    tmp = 0.0
    for i in 1:N
        # Calcolo di Sigma_g_i_f usando la funzione getSigma_g_i_f (assumiamo sia definita)
        Sigma_g_i = get_Sigma_g_i(i, tt, theta)

        # Calcolo della media mu usando la funzione get_g_hat (assumiamo sia definita)
        mu = get_mu_g(i, tt, theta)
        # Calcolo della log-pdf per la densità multivariata normale
        mvn = MvNormal(mu, Sigma_g_i)
        tmp += logpdf(mvn, theta[:g][i, :])
    end

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end



function target_gamma_k(theta, X)

    # Calcolo della log-pdf per la densità multivariata normale
    mvln = MvNormal((X * theta[:beta]), theta[:Sigma_gamma])
    tmp = logpdf(mvln, theta[:gamma])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end


function target_g_ki(i, tt, theta)
    # Calcolo di Sigma_g_i_f usando la funzione getSigma_g_i_f (assumiamo sia definita)
    Sigma_g_i = get_Sigma_g_i(i, tt, theta)

    # Calcolo della media mu usando la funzione get_g_hat (assumiamo sia definita)
    mu = get_mu_g(i, tt, theta)
    # Calcolo della log-pdf per la densità multivariata normale
    mvn = MvNormal(mu, Sigma_g_i)
    tmp = logpdf(mvn, theta[:g][i, :])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end


# function marginal_gamma_i(i, theta, Sigma_gamma, X)
#     mu = X[i,:]' * theta[:beta] + (Sigma_gamma[i, 1:end .!= i])' * inv(Sigma_gamma[1:end .!= i, 1:end .!= i])*(theta[:gamma][1:end .!= i] - X[1:end .!= i, :]*theta[:beta])
#     Sigma = Sigma_gamma[i, i] - Sigma_gamma[i, 1:end .!= i]' * inv(Sigma_gamma[1:end .!= i, 1:end .!= i]) * Sigma_gamma[1:end .!= i, i]

#     # Calcolo della log-pdf per la densità multivariata normale
#     uvn = Normal(mu, Sigma)
#     tmp = logpdf(uvn, theta[:gamma][i])

#     # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
#     out = tmp == -Inf ? -1e10 : tmp
#     return out
# end



function target_f_k(theta)
    T = size(theta[:f], 1)

    # Calcolo della log-pdf per la densità multivariata normale
    mvn = MvNormal(zeros(T), theta[:Sigma_f])
    tmp = logpdf(mvn, theta[:f])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end



function likelihood_y(y_ict, theta, sigma2_c)
    N = size(y_ict, 1)
    C = size(y_ict, 2)
    T = size(y_ict, 3)
    
    tmp = 0.0
    for c in 1:C, i in 1:N, t in 1:T
        mu = 0.0
        for k in 1:K
            mu += theta[k][:g][i,t] .* theta[k][:h][c]
        end
        uvn = Normal(mu, sqrt(sigma2_c[c]))
        tmp += logpdf(uvn, y_ict[i, c, t])
    end

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end