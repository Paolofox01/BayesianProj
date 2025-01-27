using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices

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
            g[i, k, :] = get_mu_g(i, t, f[k,:], theta_k, Sigma_f_inv)
        end

    end
    
    # Restituisce un dizionario contenente le matrici g, f, gamma
    return Dict(:g => g, :f => f, :gamma => gamma)
end


function generate_y(g, seed)
    N, K, T = size(g)          
    C = 6          
    Random.seed!(seed)         
    sigma = rand(InverseGamma(3, 2), C) 

    h = zeros(K, C)
    for k in 1:K
        h[k, :] = rand(Dirichlet(ones(C)))  # Ogni riga segue una distribuzione Dirichlet
    end

    mu = zeros(N, T, C)
    y = zeros(N, T, C)

    for i in 1:N, t in 1:T, c in 1:C
        mu[i, t, c] = sum(g[i, :, t] .* h[:, c])  # Somma data da g e h
        y[i, t, c] = rand(Normal(mu[i, t, c], sigma[c]))  # Campiona y
    end

    return y
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



# covarianza tau temporale
function tau_covariance(n, sigmatau)

    I_n = I(n)
    
    # Matrice 1 * 1^T di dimensione n x n
    ones_matrix = ones(n, n)
    
    Σ_tau = sigmatau ^2 * (I_n - (1 / (n + 1)) * ones_matrix)
    return Σ_tau
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

