from types import SimpleNamespace
import pandas as pd
import random as rand
import numpy as np
import operator
from datetime import datetime

class qlearning_struct:
    def __init__(self, S, A, Q, alpha, gamma):
        self.S = S#
        self.A = A
        self.gamma = gamma
        self.Q = Q #
        self.alpha = alpha

def max_dict(dict_struct):
    res_v = -np.infty
    res_a = None
    for a,v in dict_struct.items():
        if v > res_v:
            res_v= v
            res_a = a
    return res_v, res_a

def update(model, s, a, r, s_p):
    model.Q[s][a] += model.alpha*(r + model.gamma * max_dict(model.Q[s_p])[0] - model.Q[s][a])
    return model

def initiate_q(S,A):
    Q = {}
    for s in S:
        Q[s]={}
        for a in A:
            Q[s][a] = 0
    return Q

def main():
    total_num_states = 752 #100 #312020 #50000, 100
    file_df = pd.read_csv('/home/rptrevin/Documents/AA228/Final Project/AA228-Project/state_action_information.csv')
    S = list(range(1,total_num_states+1))
    A = set(file_df['a'].tolist())
    gamma = 0.85
    action_counts = dict(zip(list(A),[0]*len(A)))
    alpha = 0.2
    Q = initiate_q(S,A)
    model = qlearning_struct(S, A, Q, alpha, gamma)

    kmax = 50
    stats_array = np.zeros((kmax, 1))

    for i in range(kmax):
        startTime = datetime.now()

        for index, row in file_df.iterrows():
            model = update(model,row['s'], row['a'],row['r'], row['sp'])
        elapsed_time = datetime.now() - startTime
        stats_array[i, 0] = elapsed_time.total_seconds()
    print("Ave Elapsed time: " + str(np.mean(stats_array)))
    policy = {s:max_dict(Q[s]) for s in S}
    default_action = max(action_counts.items(), key=operator.itemgetter(1))[0]
    with open('/home/rptrevin/Documents/AA228/Final Project/AA228-Project/q_learning.policy', 'w') as f:
        f.write("Policy\n")
        for i in range(1, total_num_states+1):
            f.write(str(policy.get(i,default_action)[1])+"\n")
    print("Done!")

if __name__ == "__main__":
    main()