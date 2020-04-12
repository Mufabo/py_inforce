import numpy as np

def policy_evaluation(env, policy, DISC_FACTOR = 0.9, K = 10000, thresh = 10e-3):
    """
    Iteratively estiamtes the Values of all states of an environment for a specific policy
    
    Args:
        env (:obj: 'gym.environment'): The environment in which the policy is evaluated
        policy (dict:1d array[floats]): Takes state-number as keys, returns probability distribution over actions for state
        DISC_FACTOR (float): Discount factor, aka gamma. Real number in [0, 1]
        K (int): Number of max iterations
        
    Return:
        Values (1d np.array): Estimated Value under the given policy for all states of the environment
    """
    
    values = np.zeros(len(env.P.keys())) #dict.fromkeys(env.P.keys(), 0)
    values_old = np.zeros(len(env.P.keys())) #dict.fromkeys(env.P.keys(), 0)

    for k in range(K):
        # For all states s ...
        for s in env.P.keys():
            val = 0
            # ... look at all possible actions a
            for a in env.P[s].keys():
                # Assumes that the action a is an int that matches the index of its prability in the policy dict. 
                # eg policy[s][2] returns the probability of taking action 2
                #      pi(a|s)            P(s'|a, s)   reward     gamma         V(s')
                val += policy[s][a] * sum([trans[0] * (trans[2] + DISC_FACTOR * values[trans[1]]) for trans in env.P[s][a]])      

            values[s] = val
            
        #if np.linalg.norm(np.array([*values.values()]) - np.array([*values_old.values()])) < thresh:
        if np.max(np.abs(values - values_old)) < thresh:
            break
            
        values_old = np.array(values)

    return values