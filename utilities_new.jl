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
    y = zeros(Float64, n, K, n_time)

    dist = euclid_dist(sites[:, 1], sites[:, 2], n)

    for j in 1:K

        # Genera il vettore f dalla distribuzione normale multivariata con media zero e covarianza K_f
        f[j, :] = rand(MvNormal(zeros(n_time), K_f))
    
        sigma = exp.(-1 ./ theta[:rho][j] .* dist)
        
        w = rand(MvNormal(zeros(n), sigma))
    
        # Loop attraverso ciascuna colonna i
        for i in 1:n
            
            gamma_ik = get_gamma(theta[:beta][j , :], car[i, :], w[i])

            # Calcola K_i usando la funzione get_K_i (presupposta definita)
            K_i = get_K_i(x, Dict(:rho => theta[:rho_f], :tau => theta[:tau][i], :gamma => gamma_ik))
            
            # Calcola mu come K_i * K_f_inv * f
            y[i, j, :] = K_i * K_f_inv * f[j, :]

        end

    end
    
    # Restituisce un dizionario contenente le matrici y, f, z, e mu
    return Dict(:y => y, :f => f)
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
            dist[i, j] = 2 * R * asin(sqrt(a))  # Usa arcsin (asin in Julia)
            
        end
    end
    return dist
end
