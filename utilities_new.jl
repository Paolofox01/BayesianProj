using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices

#' Generate data.
#'
#' @param n Number of trials.
#' @param n_time Number of time points.
#' @param theta Named list of parameter values.
function generate_data(sites, n, K, n_time, theta)

    car = sites[:, 3:end] # caratteristiche del sito
    # Genera i valori di x
    x = range(1, stop=n_time, length=n_time)

    # Calcola K_f usando la funzione sq_exp_kernel (presupposta definita)
    K_f = sq_exp_kernel(x, theta[:rho_f]; nugget=1e-6)
    K_f_inv = inv(K_f)

    f = zeros(Float64, K, n_time)
    g = zeros(Float64, n, K, n_time)
    gamma = zeros(Float64, n, K)

    dist = euclid_dist(sites[:, 1], sites[:, 2], n)

    for j in 1:K

        # Genera il vettore f dalla distribuzione normale multivariata con media zero e covarianza K_f
        f[j, :] = rand(MvNormal(zeros(n_time), K_f))
    
        sigma = 0.5 * exp.(-1 ./ theta[:rho][j] .* dist)
        
        w = rand(MvNormal(zeros(n), sigma))
    
        # Loop attraverso ciascuna colonna i
        for i in 1:n
            
            gamma[i, j] = get_gamma(theta[:beta][j , :], car[i, :], w[i])

            # Calcola K_i usando la funzione get_K_i (presupposta definita)
            K_i = get_K_i(x, Dict(:rho => theta[:rho_f], :tau => theta[:tau][i], :gamma => gamma[i, j]))
            
            # Calcola mu come K_i * K_f_inv * f
            g[i, j, :] = K_i * K_f_inv * f[j, :]

        end

    end
    
    # Restituisce un dizionario contenente le matrici y, f, z, e mu
    return Dict(:g => g, :f => f, :gamma => gamma)
end

#' Generate covariance matrix for square exponential kernel.
#'
#' @param x Vector of time points.
#' @param rho Length scale.
#' @param alpha Amplitude.
#' @param nugget Covariance nugget.
function sq_exp_kernel(x, rho; alpha=1, nugget=0.0)
    n = length(x)
    # Calcolo dell'esponenziale quadratico
    kernel_values = alpha^2 .* exp.(-rho^2 / 2 * (x .- 1).^2)  # elementwise expo
    # Costruzione della matrice Toeplitz
    K = Matrix(Toeplitz(kernel_values, kernel_values))
    # K = Matrix(K) #la converte in una matrice di tipo normale
    # Aggiungi il nugget alla diagonale (per evitare problemi di numeri non invertibili)
    K += nugget .* I(n)   
    
    return K
end


function get_K_i(x, theta)
    n_time = length(x)
    K = Matrix{Float64}(undef, n_time, n_time)

    # Popola la matrice K
    for i in 1:n_time
        for j in 1:n_time
            K[i, j] = exp(-theta[:rho]^2 / 2 * (x[i] - x[j] - theta[:tau])^2)
        end
    end

    return theta[:gamma] .* K
end

# Calcola gamma
function get_gamma(beta::Vector{Float64}, x::Vector{Float64}, w::Float64)
    return exp(beta' * x + w)
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

#' Get y hat
#'
#' @param i Subject index (scalar, 1 < i < n).
#' @param f Vector of values for f.
#' @param theta Named list of parameter values.
#' @param K_f_inv Inverse covariance matrix of f.
function get_g_hat(i, f, theta, K_f_inv)
    # Definisce x come una sequenza di valori tra 0 e 1, con lunghezza uguale a quella di f
    x = range(1, stop=365, length=length(f))
    
    # Calcola K_i usando la funzione get_K_i (presupposta definita)
    K_i = get_K_i(x, Dict(:tau => theta[:tau][i], :gamma => theta[:beta][i], :rho => theta[:rho]))
    
    # Calcola mu come il prodotto K_i * K_f_inv * f
    mu = K_i * K_f_inv * f
    return mu
end

#' Get y_hat for all trials, output in matrix form
#'
#' @param y Matrix of observed trial data.
#' @param f Vector of f values.
#' @param theta Parameter values.
#' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
function get_g_hat_matrix(g, f, theta, K_f_inv)
    # Inizializza la matrice y_hat con il numero di righe e colonne di y
    g_hat = Matrix{Float64}(undef, size(g, 2), size(g, 1))

    # Popola ogni colonna di y_hat utilizzando la funzione get_y_hat
    for i in 1:size(g, 1)
        g_hat[:, i] = get_g_hat(i, f, theta, K_f_inv)
    end
    
    return g_hat
end

#' Get Sigma_y_i (Section 4.6.2)
#' @param beta_i Trial specific amplitude beta_i
#' @param K_f Covariance matrix of f. (n_time x n_time)
#' @param Sigma_nu Covariance AR process (n_time x n_time)
function get_Sigma_g_i(beta_i::Float64, K_f::Matrix{Float64})
    # Verifica che K_f e Sigma_nu abbiano le stesse dimensioni
    # Calcola Sigma_y_i
    r = (beta_i^2) * K_f 
    return r
end

#' Get Sigma_y_i f (Section 4.6.2)
#' @param i trial specific amplitude beta_i
#' @param x Kernel GP
#' @param theta Named list of parameter values
#' @param K_f Covariance matrix of f. (n_time x n_time).
#' @param K_f_inv Inverse covariance matrix of f (n_time x n_time).
#' @param Sigma_nu Covariance AR process (n_time x n_time).
function getSigma_g_i_f(i, x, theta, K_f, K_f_inv)
    # Calcola Sigma_y_i
    Sigma_g_i = get_Sigma_g_i(theta[:beta][i], K_f)
    
    # Calcola K_i usando get_K_i (assumendo che questa funzione sia già definita)
    K_i = get_K_i(x, Dict(:rho => theta[:rho], :tau => theta[:tau][i], :gamma => theta[:beta][i]))
    
    # Calcola Sigma_i
    Sigma_i = Sigma_g_i - (K_i)' * K_f_inv * K_i
    
    # Rendi Sigma_i simmetrica
    Sigma_i = (Sigma_i + (Sigma_i)') / 2
    
    return Sigma_i
end

#' Summary output MCMC
#' @param results output from fit_RPAGP
#' @param dat_trials simulated data in long format 
#' @param y data
#' @param burn_in burn in 
function getSummaryOutput(results, dat_trials, y, burn_in)
    n = size(y, 2)  # Numero di trials
    n_time = size(y, 1)  # Numero di time points
    n_iter = length(results.chain)  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # Ottieni le stime delle prove singole
    y_hat = getSingleTrialEstimates(results, burn_in)
    
    # Definisci i quantili
    probs = [0.025, 0.5, 0.975]
    
    # Inizializza un array per i quantili
    y_hat_quantiles = Array{Float64}(undef, 3, n_time, n)
    
    # Calcola i quantili per ogni time point e trial
    for ii in 1:n
        for t in 1:n_time
            y_hat_quantiles[:, t, ii] = quantile(y_hat[t, ii, :], probs)
        end
    end
    
    # Estrai i quantili
    lower = reshape(y_hat_quantiles[1, :, :], n_time * n, 1)
    median = reshape(y_hat_quantiles[2, :, :], n_time * n, 1)
    upper = reshape(y_hat_quantiles[3, :, :], n_time * n, 1)
    
    # Aggiungi i quantili ai dati esistenti
    out = DataFrame(dat_trials)
    out.lwr = lower
    out.med = median
    out.upr = upper
    
    return out
end


#' get singleTrialEstimates 
#' @param results output from fit_RPAGP
#' @param burn_in burn_in period
function getSingleTrialEstimates(results, burn_in)
    n_iter = length(results.chain)  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # - Estrazione di beta
    chain_beta_burned = zeros(n, n_final)  # Matrice per beta
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_beta_burned[:, ss] = results.chain[tt][:beta]  # Beta dalla catena
        ss += 1
    end
    
    # - Estrazione di f
    chain_f_burned = zeros(n_time, n_final)  # Matrice per f
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_f_burned[:, ss] = results.chain_f[tt]  # f dalla catena
        ss += 1
    end
    
    # - Calcolo di y_hat
    y_hat = zeros(n_time, n, n_final)  # Array per le stime
    for tt in 1:n_final
        for ii in 1:n
            y_hat[:, ii, tt] = chain_beta_burned[ii, tt] * chain_f_burned[:, tt]
        end
    end
    
    return y_hat
end


#' Transform an upper triangular matrix to symmetric
#' @param m upper triangular matrix
function ultosymmetric(m)
    m = m + transpose(m) - Diagonal(Diagonal(m))
    return m
end

