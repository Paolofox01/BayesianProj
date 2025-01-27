using Distributions, LinearAlgebra

include("utilities.jl")

function likelihood(t, g, f, theta, Sigma_f, Sigma_f_inv)
    n = size(g, 1)
    n_time = size(g, 2)
    
    tmp = 0.0
    for i in 1:n
        # Calcolo di Sigma_g_i_f usando la funzione getSigma_g_i_f (assumiamo sia definita)
        Sigma_g_i = get_Sigma_g_i(i, t, theta, Sigma_f, Sigma_f_inv)

       # Sigma_g_i_f = (Sigma_g_i_f + (Sigma_g_i_f)') / 2  # Rendere la matrice simmetrica

        # Calcolo della media mu usando la funzione get_g_hat (assumiamo sia definita)
        mu = get_mu_g(i, t, f, theta, Sigma_f_inv)
        # Calcolo della log-pdf per la densità multivariata normale
        mvn = MvNormal(mu, Sigma_g_i)
        tmp += logpdf(mvn, g[i, :])
        # println(i)
    end

    
    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end


function target_gamma(theta, Sigma_gamma, X)
    # n = size(gamma, 1)

    # Calcolo della log-pdf per la densità multivariata normale
    mvln = MvNormal((X * theta[:beta]), Sigma_gamma)
    tmp = logpdf(mvln, theta[:gamma])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end


function target_g_i(i, t, g, f, theta, Sigma_f, Sigma_f_inv)
    # Calcolo di Sigma_g_i_f usando la funzione getSigma_g_i_f (assumiamo sia definita)
    Sigma_g_i = get_Sigma_g_i(i, t, theta, Sigma_f, Sigma_f_inv)

    # Calcolo della media mu usando la funzione get_g_hat (assumiamo sia definita)
    mu = get_mu_g(i, t, f, theta, Sigma_f_inv)
    # Calcolo della log-pdf per la densità multivariata normale
    mvn = MvNormal(mu, Sigma_g_i)
    tmp = logpdf(mvn, g[i, :])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end


function marginal_gamma_i(i, theta, Sigma_gamma, X)
    mu = X[i,:]' * theta[:beta] + (Sigma_gamma[i, 1:end .!= i])' * inv(Sigma_gamma[1:end .!= i, 1:end .!= i])*(theta[:gamma][1:end .!= i] - X[1:end .!= i, :]*theta[:beta])
    Sigma = Sigma_gamma[i, i] - Sigma_gamma[i, 1:end .!= i]' * inv(Sigma_gamma[1:end .!= i, 1:end .!= i]) * Sigma_gamma[1:end .!= i, i]

    # Calcolo della log-pdf per la densità multivariata normale
    mvln = Normal(mu, Sigma)
    tmp = logpdf(mvln, theta[:gamma][i])

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end