using Distributions, LinearAlgebra, Random, LogExpFunctions

include("utilities.jl")
include("proposal_functions.jl")



function sample_tau_k(k, tt, current, hyperparam)
    curr = deepcopy(current[k])
    N = length(curr[:tau])

    for i in 1:N
        proposed = deepcopy(curr)
        proposed[:tau][i] = propose_tau_i(curr[:tau][i], hyperparam)

        lik_current = target_g_i(i, tt, curr)
        prior_current = prior[:tau_i](curr[:tau][i], hyperparam)
        
        lik_proposed = target_g_i(i, tt, proposed)
        prior_proposed = prior[:tau_i](proposed[:tau][i], hyperparam)
        
        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
        if prob > rand()
            curr[:tau][i] = deepcopy(proposed[:tau][i])
        end
    end

    return curr[:tau]
end




function sample_rho_k(k, tt, current, hyperparam)
    T = length(tt)
    
    proposed = deepcopy(current[k])
    proposed[:rho] = exp( propose_log_rho(log(current[k][:rho]), hyperparam) )
    proposed[:Sigma_f] = sq_exp_kernel(tt, proposed[:rho])
    #Sigma_f_prop_inv = inv(Sigma_f_prop)
    proposed[:Sigma_f_inv] = trench(proposed[:Sigma_f])

    
    lik_current = likelihood(tt, current) + target_f(current[:f], current[:Sigma_f])
    prior_current = prior[:rho](current[k][:rho], hyperparam) + log(current[k][:rho])
    
    lik_proposed = likelihood(tt, proposed) + target_f(proposed[:f], proposed[:Sigma_f])
    prior_proposed = prior[:rho](proposed[:rho], hyperparam) + log(proposed[:rho])
    
    prob = exp( lik_proposed + prior_proposed  - lik_current - prior_current )
    
    if prob > rand()
        return proposed[:rho], proposed[:Sigma_f], proposed[:Sigma_f_inv]
    else
        return current[:rho], current[:Sigma_f], current[:Sigma_f_inv]
    end
end




function sample_f(tt, theta)
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




function sample_gamma(tt, current, X)
    N = size(g, 1)
    curr = deepcopy(current)

    for i in 1:N
        proposed = deepcopy(curr)
        proposed[:gamma][i] = propose_gamma_i(curr[:gamma][i], hyperparam)
        

        lik_current = target_g_i(i, t, g, f, curr, Sigma_f, Sigma_f_inv)
        #prior_current = marginal_gamma_i(i, curr, Sigma_gamma, X)
        prior_current = target_gamma(curr, Sigma_gamma, X)
        
        lik_proposed = target_g_i(i, t, g, f, proposed, Sigma_f, Sigma_f_inv)
        #prior_proposed = marginal_gamma_i(i, proposed, Sigma_gamma, X)
        prior_proposed = target_gamma(proposed, Sigma_gamma, X)

        prob = exp(lik_proposed + prior_proposed - lik_current - prior_current)
        if prob > rand()
            curr[:gamma][i] = deepcopy(proposed[:gamma][i])
        end
    end

    return curr[:gamma]
end




function sample_beta(current, Sigma_gamma, X)
    inv_S = inv(Sigma_gamma)
    S = inv(X' * inv_S * X + I(size(X,2)))
    S = (S+S')/2
    m = S' * X' * inv_S' * current[:gamma]

    return rand(MultivariateNormal(m, S))
end




function sample_phi(D, X, current, hyperparam)
    
    proposed = deepcopy(current)
    proposed[:phi] = exp( propose_log_phi(log(current[:phi]), hyperparam) )
    
    Sigma_gamma_prop = get_Sigma_gamma(D, proposed[:phi])
    Sigma_gamma_curr = get_Sigma_gamma(D, current[:phi])
    
    lik_current = target_gamma(current, Sigma_gamma_curr, X)
    prior_current = prior[:phi](current[:phi], hyperparam) + log(current[:phi])
    
    lik_proposed = target_gamma(proposed, Sigma_gamma_prop, X) 
    prior_proposed = prior[:phi](proposed[:phi], hyperparam) + log(proposed[:phi])
    
    prob = exp(lik_proposed + prior_proposed - lik_current - prior_current) 
    
    if prob > rand()
        return proposed[:phi], Sigma_gamma_prop
    else
        return current[:phi], Sigma_gamma_curr
    end
end




function sample_sigma_c(c, h, g, y_ict, hyperparam)
    N = size(g, 1)
    T = size(g, 3)

    tmp = 0.0
    for i in 1:N, t in 1:T
        tmp +=  y_ict[i,c,t] - g[i,:,t]' * h[c,:]
    end
    b_new = hyperparam[:prior_sc_b] + 0.5 * tmp
    a_new = hyperparam[:prior_sc_a] + (N*T/2)

    return rand(InverseGamma(a_new, b_new))
end




function sample_h_k(k, y_ict, g, h, sigma_c, hyperparam)
    C = size(h,1)
    curr_alr = log.(h[1:(C-1),k]./h[C,k])
    prop_alr = propose_h_alr(h[:,k], h_proposal_sd)

    h_proposed = deepcopy(h)
    h_proposed[:,k] = softmax([prop_alr; 0])
    
    lik_current = likelihood_y(y_ict, g, h, sigma_c)
    prior_current = prior[:h](h[:,k], hyperparam) + sum(curr_alr) - C * logsumexp([curr_alr; 0])
    
    lik_proposed = likelihood_y(y_ict, g, h_proposed, sigma_c)
    prior_proposed = prior[:h](h_proposed[:,k], hyperparam) + sum(prop_alr) - C * logsumexp([prop_alr; 0])

    prob = exp(lik_proposed + prior_proposed  - lik_current - prior_current ) 
    
    if prob > rand()
        return h_proposed 
    else
        return h
    end
end




function sample_g_ik(i, k, tt, y_ict, current, sigma_c)
    T = length(t)
    inv_Sg = inv( get_Sigma_g_i(i, t, current, Sigma_f, Sigma_f_inv) )
    S = inv( sum(h[:,k].^2 ./ sigma_c[1:C]).*I(T) + inv_Sg )
    S = (S+S')/2
    m = inv_Sg' * get_mu_g(i,t,f,current,Sigma_f_inv) 
    for c in 1:C
        m += sum( h[c,k] / sigma_c[c] .* (y_ict[i,c,:] - g[i,:,:].*h[:,c]) )
    end
    m = S' * m
    return rand(MultivariateNormal(m, S))
end