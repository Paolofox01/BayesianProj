using Distributions, LinearAlgebra

include("utilities_new.jl")

function likelihood(g, f, theta, K_f, K_f_inv)
    n = size(g, 1)
    n_time = size(g, 2)
    x = range(1, stop=365, length=n_time)
    
    tmp = 0.0
    for i in 1:n
        # Calcolo di Sigma_y_i_f usando la funzione getSigma_y_i_f (assumiamo sia definita)
        Sigma_g_i_f = getSigma_g_i_f(i, x, theta, K_f, K_f_inv)

       # Sigma_g_i_f = (Sigma_g_i_f + (Sigma_g_i_f)') / 2  # Rendere la matrice simmetrica

       # println(det(Sigma_g_i_f))

        # Calcolo della media mu usando la funzione get_y_hat (assumiamo sia definita)
        mu = get_g_hat(i, f, theta, K_f_inv)
        mu = mu[:, 1]
        # Calcolo della log-pdf per la densità multivariata normale
        mvn = MvNormal(mu, Sigma_g_i_f)
        tmp += logpdf(mvn, g[i, :])

        # println(i)
    end

    

    # Restituisce -1e10 se il risultato è -Inf, altrimenti restituisce tmp
    out = tmp == -Inf ? -1e10 : tmp
    return out
end