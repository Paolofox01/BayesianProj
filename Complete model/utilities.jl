using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices

#' Generate data.
#'
#' @param n Number of trials.
#' @param n_time Number of time points.
#' @param theta Named list of parameter values.
function generate_data(sites, n, K, n_time, theta)

    X = sites[:, 3:end] # Design matrix: caratteristiche del sito
    coords = sites[:, 1:2] # coordinate geografiche
    
    # Genera i valori di t
    t = range(1, stop=n_time, length=n_time)

    f = zeros(Float64, K, n_time)
    g = zeros(Float64, n, K, n_time)
    gamma = zeros(Float64, n, K)

    dist = euclid_dist(coords[:, 1], coords[:, 2], n)

    for k in 1:K
        theta_k = theta[k]
        # Calcola Sigma_f usando la funzione sq_exp_kernel (presupposta definita)
        Sigma_f = sq_exp_kernel(t, theta_k[:rho]; nugget=1e-6)
        Sigma_f_inv = inv(Sigma_f)
        # Genera il vettore f dalla distribuzione normale multivariata con media zero e matrice varianza Sigma_f
        f[k, :] = rand(MvNormal(zeros(n_time), Sigma_f))
         
        Sigma_gamma = get_Sigma_gamma(dist, theta_k[:phi])
        gamma[:, k] = rand(MvNormal(X*theta_k[:beta], Sigma_gamma))
        theta_k[:gamma] = gamma[:, k]
        # Loop attraverso ciascuna colonna i
        for i in 1:n
            # Calcola Sigma_i usando la funzione get_Sigma_i (presupposta definita)
            Sigma_i = get_Sigma_i(i, t, theta_k)

            # Calcola mu come Sigma_i * Sigma_f_inv * f
            g[i, k, :] = rand(MultivariateNormal(get_mu_g(i, t, f[k,:], theta_k, Sigma_f_inv) , get_Sigma_g_i(i,t,theta_k,Sigma_f,Sigma_f_inv)))
        end

    end
    
    # Restituisce un dizionario contenente le matrici g, f, gamma
    return Dict(:g => g, :f => f, :gamma => gamma)
end


#' Generate covariance matrix for square exponential kernel.
#'
#' @param t Vector of time points.
#' @param rho Length scale.
#' @param alpha Amplitude.
#' @param nugget Covariance nugget.
function sq_exp_kernel(t, rho; alpha=1, nugget=0.0)
    n_time = length(t)
    # Calcolo dell'esponenziale quadratico
    K = Matrix{Float64}(undef, n_time, n_time)

    # Popola la matrice K
    for i in 1:n_time
        for j in 1:n_time
            K[i, j] = alpha^2 * exp(- (rho)^2 / 2 * (t[i] - t[j])^2)
        end
    end
    
    K += nugget .* I(n_time)   
    K = (K + K')/2
    
    return K
end



#' Generate covariance matrix for square exponential kernel.
#'
#' @param D matrix of Euclidean distances.
#' @param phi Length scale.
#' @param alpha Amplitude.
#' @param nugget Covariance nugget.
function get_Sigma_gamma(D, phi; alpha=1, nugget=0.1)
    n_stations = size(D, 1)
    # Calcolo dell'esponenziale quadratico
    K = Matrix{Float64}(undef, n_stations, n_stations)

    # Popola la matrice K
    for i in 1:n_stations
        for j in 1:n_stations
            K[i, j] = alpha^2 * exp(-phi^2 / 2 * (D[i,j])^2)
        end
    end
    
    K += nugget .* I(n_stations)   
    K = (K + K')/2
    
    #println(isposdef(K))

    return K
end




function get_Sigma_i(i, t, theta)
    n_time = length(t)
    K = Matrix{Float64}(undef, n_time, n_time)

    # Popola la matrice K
    for ii in 1:n_time
        for jj in 1:n_time
            K[ii, jj] = exp(-theta[:rho]^2 / 2 * (t[ii] - t[jj] - theta[:tau][i])^2)
        end
    end

    #return theta[:gamma] .* K
    return K
end






# Calcolo della distanza euclidea
function euclid_dist(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    
    R = 6371.0
    dist = zeros(Float64, n, n)

    for i in 1:n
        for j in 1:n
            lat1, lon1 = deg2rad(x[i]), deg2rad(y[i])
            lat2, lon2 = deg2rad(x[j]), deg2rad(y[j])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
            dist[i, j] = 2 * R * asin(min(1, sqrt(a)))  # Usa arcsin (asin in Julia)
            
        end
    end
    return dist
end





#' Get g_ki mean
#'
#' @param i Subject index (scalar, 1 < i < n).
#' @param t Vector of time instances.
#' @param f Vector of values for f.
#' @param theta Named list of parameter values.
#' @param gamma Vector of values for gamma.
#' @param Sigma_f_inv Inverse covariance matrix of f.
function get_mu_g(i, t, f, theta, Sigma_f_inv)
    
    # Calcola Sigma_i usando la funzione get_Sigma_i (presupposta definita)
    Sigma_i = get_Sigma_i(i, t, theta)
    
    # Calcola mu come il prodotto Sigma_i * Sigma_f_inv * f
    mu = exp(theta[:gamma][i]) * Sigma_i * Sigma_f_inv * f
    return mu[:,1]
end



#' Get mu_g for all trials, output in matrix form
#'
#' @param g Matrix of observed trial data.
#' @param f Vector of f values.
#' @param theta Parameter values.
#' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
function get_mu_g_matrix(g, f, t, theta, Sigma_f_inv)
    # Inizializza la matrice y_hat con il numero di righe e colonne di y
    g_hat = Matrix{Float64}(undef, size(g, 1), size(g, 2))

    # Popola ogni colonna di y_hat utilizzando la funzione get_mu_g
    for i in 1:size(g, 1)
        g_hat[i, :] = get_mu_g(i, t, f, theta, Sigma_f_inv)
    end
    
    return g_hat
end



#' Get Sigma_g_i (Section 4.6.2)
#' @param beta_i Trial specific amplitude beta_i
#' @param Sigma_f Covariance matrix of f. (n_time x n_time)
#' @param Sigma_nu Covariance AR process (n_time x n_time)
function get_Sigma_g_i(i, t, theta, Sigma_f, Sigma_f_inv)

    Sigma_i = get_Sigma_i(i, t, theta)

    K = (exp(theta[:gamma][i])^2) .* (Sigma_f - Sigma_i * Sigma_f_inv * (Sigma_i)' ) + 0.01 .* I(size(Sigma_f, 1))
    # Simmetrizzo
    K  = (K  + K') / 2

    #println("staz: ", i, "posdef:", isposdef(K))
    #println("staz: ", i, "symmetric:", issymmetric(K))
    return K
end


#' Get Sigma_y_i f (Section 4.6.2)
#' @param i trial specific amplitude beta_i
#' @param x Kernel GP
#' @param theta Named list of parameter values
#' @param Sigma_f Covariance matrix of f. (n_time x n_time).
#' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
#' @param Sigma_nu Covariance AR process (n_time x n_time).
# function getSigma_g_i_f(i, x, theta, Sigma_f, Sigma_f_inv)
#     # Calcola Sigma_y_i
#     Sigma_g_i = get_Sigma_g_i(theta[:gamma][i], Sigma_f)
    
#     # Calcola Sigma_i usando get_Sigma_i (assumendo che questa funzione sia già definita)
#     Sigma_i = get_Sigma_i(x, Dict(:rho => theta[:rho], :tau => theta[:tau][i], :gamma => theta[:gamma][i]))

#     # Calcola Sigma_i
#     Sigma_i = Sigma_g_i - (Sigma_i)' * Sigma_f_inv * Sigma_i
    
#     # Rendi Sigma_i simmetrica
#     Sigma_i = (Sigma_i + (Sigma_i)') / 2
    
#     return Sigma_i
# end




# function get_sigma_loggamma_g(n, n_time, f, Sigma_gamma, mu_loggamma, mu_gamma, current, Sigma_f_inv)

#     x = range(1, stop=n_time, length=n_time)

#     f_matrix = zeros(n, n * n_time)

#     mu_f_tau = zeros(n_time) #da modificare se GP troncato

#     mu_f = zeros(n_time) #da modificare se GP troncato

#     for i in 1:n

#         Sigma_f_tau_i_and_f = get_Sigma_i(x, Dict(:rho => current[:rho], :tau => current[:tau][i], :gamma => 1.0))

#         mu_f_tau_i_dato_f = mu_f_tau + Sigma_f_tau_i_and_f * Sigma_f_inv * (f - mu_f)

#         f_matrix[i, ((i-1)*n_time)+1:i*n_time] = mu_f_tau_i_dato_f
#     end

#     loggamma_gamma_matrix = zeros(n, n)

#     for i in 1:n
#         for j in 1:n

#             loggamma_gamma_matrix[i,j] = exp(mu_loggamma[j] + Sigma_gamma[j,j]/2) * (mu_loggamma[i] + Sigma_gamma[i,j])
            
#         end
#     end

#     return loggamma_gamma_matrix * f_matrix - mu_loggamma * mu_gamma' * f_matrix, f_matrix', f_matrix' * mu_gamma


# end




#' Summary output MCMC
#' @param results output from fit_RPAGP
#' @param dat_trials simulated data in long format 
#' @param y data
#' @param burn_in burn in 
function getSummaryOutput(results, dat_trials, g, burn_in)
    n = size(g, 1)  # Numero di trials
    n_time = size(g, 2)  # Numero di time points
    n_iter = length(results[:chain])  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # Ottieni le stime delle prove singole
    GSTE = getSingleTrialEstimates(results, burn_in, n, n_time)
    
    g_hat = GSTE[:g_hat]

    f_hat = GSTE[:chain_f_burned]

    # Definisci i quantili
    probs = [0.025, 0.5, 0.975]
    
    # Inizializza un array per i quantili
    g_hat_quantiles = Array{Float64}(undef, 3, n, n_time)
    
    # Calcola i quantili per ogni time point e trial
    for ii in 1:n
        for t in 1:n_time
            g_hat_quantiles[:, ii, t] = quantile(g_hat[ii, t, :], probs)
        end
    end
    
    # Estrai i quantili
    lower = Vector(reshape(g_hat_quantiles[1, :, :], n_time * n))
    median = Vector(reshape(g_hat_quantiles[2, :, :], n_time * n))
    upper = Vector(reshape(g_hat_quantiles[3, :, :], n_time * n))
    mean_f= vec(mean(f_hat, dims=2))

    # Aggiungi i quantili ai dati esistenti
    out = DataFrame(dat_trials)
    out.lwr = lower
    out.med = median
    out.upr = upper

    
    return out, mean_f, g_hat_quantiles
end




#' get singleTrialEstimates 
#' @param results output from fit_RPAGP
#' @param burn_in burn_in period
function getSingleTrialEstimates(results, burn_in, n, n_time)
    n_iter = length(results[:chain])  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # - Estrazione di beta
    chain_gamma_burned = zeros(n, n_final)  # Matrice per beta
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_gamma_burned[:, ss] = results[:chain][tt][:gamma]  # gamma dalla catena
        ss += 1
    end
    
    # - Estrazione di f
    chain_f_burned = zeros(n_time, n_final)  # Matrice per f
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_f_burned[:, ss] = results[:chain_f][tt]  # f dalla catena
        ss += 1
    end
    
    # - Calcolo di y_hat
    g_hat = zeros(n, n_time, n_final)  # Array per le stime
    for tt in 1:n_final
        for ii in 1:n
            g_hat[ii, :, tt] = chain_gamma_burned[ii, tt] * chain_f_burned[:, tt]
        end
    end
    
    return Dict(:g_hat => g_hat, :chain_f_burned => chain_f_burned)
end



#' Transform an upper triangular matrix to symmetric
#' @param m upper triangular matrix
function ultosymmetric(m)
    m = m + transpose(m) - Diagonal(Diagonal(m))
    return m
end

