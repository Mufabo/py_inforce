import numpy as np
import py_inforce as pin

def policy_iteration(env, DISC_FACTOR = 0.9, K= 10000, thresh=10e-3):
    """
    Returns a new policy that acts greedily on the values
    
    Args:
        env (gym environment):
        
        
    return:
        policy (2darray[float]): mapping states given as ints to arrays containing the probability distribution over all actions
    """
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_old = np.ones([env.nS, env.nA]) / env.nA
    converged = False
    while not converged:
        # Evaluate current policy
        values = pin.policy_evaluation(env, policy, DISC_FACTOR, K, thresh)
        Q = np.zeros((env.nS, env.nA)) # Q(s, a)
        for s in env.P.keys():
            # One-step look-ahead            
            for a in env.P[s].keys():
                Q[s, a] = sum([trans[0] * (trans[2] + DISC_FACTOR * values[trans[1]]) for trans in env.P[s][a] ])
            
            policy[s, :] = 0
            policy[s, np.argmax(Q[s, :])] = 1
            if (policy[s, :] == policy_old[s, :]).all():
                converged = True
            
        policy_old = np.array(policy)

    return policy
    