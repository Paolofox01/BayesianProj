using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices


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
    
        sigma = exp.(-1 ./ theta[:rho][j] .* dist)
        
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


function sq_exp_kernel(x, rho; alpha=1, nugget=0.0)
    n_time = length(x)
    # Calcolo dell'esponenziale quadratico
    K = Matrix{Float64}(undef, n_time, n_time)

    # Popola la matrice K
    for i in 1:n_time
        for j in 1:n_time
            K[i, j] = alpha^2 * exp(-rho^2 / 2 * (x[i] - x[j])^2)
        end
    end

    K += nugget .* I(n_time)   
    
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


function generate_y(g)
    N, K, T = size(g)          
    C = 6                      
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