using Distributions, LinearAlgebra, Random, LogExpFunctions

include("utilities.jl")
include("proposal_functions.jl")
include("target_functions.jl")



function sample_tau_k(tt, current, hyperparam)
    curr = deepcopy(current)
    N = length(curr[:tau])

    for i in 1:N
        proposed = deepcopy(curr)
        proposed[:tau][i] = propose_tau_i(curr[:tau][i], hyperparam)

        lik_current = target_g_ki(i, tt, curr)
        prior_current = prior[:tau_i](curr[:tau][i], hyperparam)
        
        lik_proposed = target_g_ki(i, tt, proposed)
        prior_proposed = prior[:tau_i](proposed[:tau][i], hyperparam)
        
        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
        if prob > rand()
            curr[:tau][i] = deepcopy(proposed[:tau][i])
        end
    end

    return curr[:tau]
end




function sample_rho_k(tt, current, hyperparam)
    T = length(tt)
    
    proposed = deepcopy(current)
    proposed[:rho] = exp( propose_log_rho(log(current[:rho]), hyperparam) )
    proposed[:Sigma_f] = sq_exp_kernel(tt, proposed[:rho])
    #Sigma_f_prop_inv = inv(Sigma_f_prop)
    proposed[:Sigma_f_inv] = trench(proposed[:Sigma_f])

    
    lik_current = target_g_k_prod(tt, current) + target_f_k(current)
    prior_current = prior[:rho](current[:rho], hyperparam) + log(current[:rho])
    
    lik_proposed = target_g_k_prod(tt, proposed) + target_f_k(proposed)
    prior_proposed = prior[:rho](proposed[:rho], hyperparam) + log(proposed[:rho])
    
    prob = exp( lik_proposed + prior_proposed  - lik_current - prior_current )
    
    if prob > rand()
        return proposed[:rho], proposed[:Sigma_f], proposed[:Sigma_f_inv]
    else
        return current[:rho], current[:Sigma_f], current[:Sigma_f_inv]
    end
end




function sample_f_k(tt, theta)
    N = size(theta[:g], 1)
    T = size(theta[:g], 2)
    
    A = theta[:Sigma_f_inv]
    b = zeros(T)
        
    for i in 1:N
        Sigma_g_i = get_Sigma_g_i(i, tt, theta)
        Sigma_i = get_Sigma_i(i, tt, theta)
        
        L = Sigma_i * theta[:Sigma_f_inv]
        G = inv(Sigma_g_i) * L
        A += L' * G
        b += (theta[:g][i, :]' * G)[:]
    end

    Sigma_f_post = inv(A)
    Sigma_f_post = (Sigma_f_post + Sigma_f_post') / 2
    
    return rand(MultivariateNormal(Sigma_f_post * b, Sigma_f_post))
end




function sample_gamma_k(tt, current, X)
    N = size(X, 1)
    curr = deepcopy(current)

    for i in 1:N
        proposed = deepcopy(curr)
        proposed[:gamma][i] = propose_gamma_i(curr[:gamma][i], hyperparam)
        

        lik_current = target_g_ki(i, tt, curr)
        #prior_current = marginal_gamma_i(i, curr, Sigma_gamma, X)
        prior_current = target_gamma_k(curr, X)
        
        lik_proposed = target_g_ki(i, tt, proposed)
        #prior_proposed = marginal_gamma_i(i, proposed, Sigma_gamma, X)
        prior_proposed = target_gamma_k(proposed, X)

        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
        if prob > rand()
            curr[:gamma][i] = deepcopy(proposed[:gamma][i])
        end
    end

    return curr[:gamma]
end




function sample_beta_k(current, X)
    inv_S = inv(current[:Sigma_gamma])
    S = inv(X' * inv_S * X + I(size(X,2)))
    S = (S+S')/2
    m = S' * X' * inv_S' * current[:gamma]

    return rand(MultivariateNormal(m, S))
end




function sample_phi_k(D, X, current, hyperparam)
    
    proposed = deepcopy(current)
    proposed[:phi] = exp( propose_log_phi(log(current[:phi]), hyperparam) )
    
    Sigma_gamma_prop = get_Sigma_gamma(D, proposed[:phi])
    Sigma_gamma_curr = get_Sigma_gamma(D, current[:phi])
    
    lik_current = target_gamma_k(current, X)
    prior_current = prior[:phi](current[:phi], hyperparam) + log(current[:phi])
    
    lik_proposed = target_gamma_k(proposed, X) 
    prior_proposed = prior[:phi](proposed[:phi], hyperparam) + log(proposed[:phi])
    
    prob = exp(lik_proposed + prior_proposed - lik_current - prior_current) 
    
    if prob > rand()
        return proposed[:phi], Sigma_gamma_prop
    else
        return current[:phi], Sigma_gamma_curr
    end
end




function sample_sigma2_c(c, current, y_ict, hyperparam)
    N = size(y_ict, 1)
    T = size(y_ict, 3)
    K = length(current)     #number of entries in the dictionary

    tmp = 0.0
    for i in 1:N, t in 1:T
        y_mean = 0.0
        for k in 1:K
            y_mean += current[k][:g][i,t] * current[k][:h][c]
        end
        tmp +=  y_ict[i,c,t] -  y_mean
    end
    b_new = hyperparam[:prior_sc_b] + 0.5 * tmp
    a_new = hyperparam[:prior_sc_a] + (N*T/2)

    return rand(InverseGamma(a_new, b_new))
end




function sample_h_k(k, y_ict, sigma_c, current, hyperparam)
    C = size(current[k][:h],1)
    curr_alr = log.(current[k][:h][1:(C-1)]./current[k][:h][C])
    prop_alr = propose_h_alr(current[k][:h], hyperparam)

    proposed = deepcopy(current)
    proposed[k][:h] = softmax([prop_alr; 0])
    
    lik_current = likelihood_y(y_ict, current, sigma_c)
    prior_current = prior[:h](current[k][:h], hyperparam) + sum(curr_alr) - C * logsumexp([curr_alr; 0])
    
    lik_proposed = likelihood_y(y_ict, proposed, sigma_c)
    prior_proposed = prior[:h](proposed[k][:h], hyperparam) + sum(prop_alr) - C * logsumexp([prop_alr; 0])

    prob = exp(lik_proposed + prior_proposed  - lik_current - prior_current) 
    
    if prob > rand()
        return proposed[k][:h] 
    else
        return current[k][:h]
    end
end




function sample_g_ik(i, k, tt, y_ict, current, sigma_c)
    T = length(tt)
    C = length(current[k][:h])
    inv_Sg = inv( get_Sigma_g_i(i, tt, current[k]) )
    S = inv( sum(current[k][:h].^2 ./ sigma_c[1:C].^2).*I(T) + inv_Sg )
    S = (S+S')/2
    m = inv_Sg' * get_mu_g(i, tt, current[k]) 
    for c in 1:C
        y_mean = zeros(T)
        for kk in (1:K)[Not(k)]
            y_mean += current[kk][:g][i,:] .* current[kk][:h][c]
        end
        m += current[k][:h][c] / sigma_c[c]^2 .* (y_ict[i,c,:] - y_mean)
    end
    m = S' * m
    return rand(MultivariateNormal(m, S))
end



function sample_y_post(current, sigma2_c, N, C, T)
    y_out = zeros(Float64, N, C, T)
    for c in 1:C, i in 1:N
        y_mean = zeros(T)
        for k in 1:K
            y_mean += current[k][:g][i,:] .* current[k][:h][c]
        end
        y_out[i,c,:] = rand(MultivariateNormal(y_mean, sigma2_c[c] .* I(T)))
    end

    return y_out
end