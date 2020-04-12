import numpy as np

def value_iteration(env, DISC_FACTOR = 0.9, thresh=10e-4):
    """
    Args:
        env (gym environment)
        
    Returns:
        values
        policy
    """
    values = np.zeros(env.nS)
    
    while True:
        tmp = np.array(values)
        for s in env.P.keys():
            
            
            Q_s = np.zeros(len(env.P[s].keys()))
            for a in env.P[s].keys():
                Q_s[a] = sum([trans[0] * (trans[2] + DISC_FACTOR * values[trans[1]]) for trans in env.P[s][a]])
            values[s] = max(Q_s)
    
            
        if np.max(np.abs(tmp - values)) < thresh:
            break
    
    policy = np.zeros([env.nS, env.nA])
    for s in env.P.keys():
        Q_s = np.zeros(len(env.P[s].keys()))
        for a in env.P[s].keys():
            Q_s[a] = sum([trans[0] * (trans[2] + DISC_FACTOR * values[trans[1]]) for trans in env.P[s][a]])
            
        policy[s, np.argmax(Q_s)] = 1
        
    return policy, values