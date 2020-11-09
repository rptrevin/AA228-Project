from types import SimpleNamespace
import pandas as pd
import random as rand
import numpy as np
import operator
from datetime import datetime

class MDP_struct:
    def __init__(self, gamma, A, S, T, R, TR):
        self.gamma = gamma
        self.A = A
        self.S = S
        self.T = T
        self.R = R
        self.TR = TR

class ValueIteration:
    def __init__(self, k_max,):
        self.k_max = k_max # maximum number of iterations

class ValueFunctionPolicy:
    def __init__(self, P, U):
        self.P = P # problem
        self.U = U # utility function


def solve(M, P):
    U = [0.0 for s in P.S ]
    for k in range(M.k_max):
        U_p = [backup( P , U, s) for s in P.S ]
        U = U_p

    return ValueFunctionPolicy(P , U)

def sum_row(row_dict):
    res = 0
    for key, val  in row_dict.items():
        res += val
    return res



def solve_value_iteration(M_val_iter, model):
    U = {s:0.0 for s in model.S }
    for k in range(1, M_val_iter.k_max):
        U_p = {s:backup( model, U, s) for s in model.S }
        U = U_p

    P = MDP(model)
    return ValueFunctionPolicy(P, U)



def update(planner, model, s, a, r, s_p):
    P = MDP_struct(model)
    U = solve(P).U
    model.U = U.copy()
    return planner

def backup(model, U, s):
    return max([lookahead(model, s, a) for a in model.A])

def MDP(N, rho, S, A, gamma):

    T = N.copy()
    R = rho.copy()
    for s in S:
        for a in A:
            e_flag = False
            n = 0
            if s in N.keys():
                if a in N[s].keys():
                    n = sum_row(N[s][a])
                    e_flag = True

            if n == 0:
                if e_flag:
                    for sp in N[s][a].keys():
                        T[s][a][sp] = 0.0
                    R[s][a] = 0.0
            else:
                for sp in N[s][a].keys():
                    T[s][a][sp] = N[s][a][sp] / n

                R[s][a] = float(rho[s][a]) / (n)
    return MDP_struct(gamma, A, S, T, R, None)


def lookahead( P, U, s, a):
    S,T,R,gamma = P.S, P.T, P.R, P.gamma
    if s in T.keys():
        if a in T[s].keys():
            if s in R.keys():
                return R[s][a] + gamma * sum([T[s][a][s_p] * U[s_p] for s_p in T[s][a].keys()])
        else:
            return 0
    else:
        if s in R.keys():
            if a in R[s].keys():
                return R[s][a]
            else:
                return 0
        else:
            return 0

def _findmax(list_struct):
    prev_v = -np.infty
    prev_a = None
    for i, (v, a) in enumerate(list_struct):
        if v > prev_v:
            prev_v = v
            prev_a = a
    return (prev_v, prev_a)

def greedy(P, U, s):
    u, a = _findmax([(lookahead(P, U, s, a), a) for a in P.A])
    return (a, u)

def valuefunctionpolicy(P, U):
    return {s:greedy(P, U, s)[0] for s in P.S}

def update_full(planner, model, s, a, r, s_p):
    U = solve_value_iteration(planner, model).U
    model.U = U.copy()
    return planner

def create_dirichlet_dict(total_num_state, N, A):
    D = {}
    for i in range(1,total_num_state+1):
        for a in A:
            if i in D.keys():
                D[i][a] = 1
            else:
                D[i]={a: 1}

            if i in N.keys():
                if a in N[i].keys():
                    D[i][a] += sum_row(N[i][a])
    return D

class PolicyIteration:
    def __init__(self, pi, k_max):
        self.pi = pi # initial policy
        self.k_max = k_max # maximum number of iterations



def iterative_policy_evaluation( P, pi, k_max):
    S, T, R, gamma = P.S, P.T, P.R, P.gamma
    U = {s:0.0 for s in S }
    for k in range(1,k_max+1):
        U = {s:lookahead( P, U, s, pi[s]) for s in S}
    return U

def all(pi, pi_p, S):
    for s in S:
        if pi_p[s] != pi[s]:
            return False
    return True

def solve_policy_iteration(M, P ):
    pi, S = M.pi, P.S
    converged = False
    for k in range(1,M.k_max+1):
        U = iterative_policy_evaluation(P, pi, M.k_max)
        pi_p = valuefunctionpolicy( P , U)
        if all(pi, pi_p, S):
            break

        pi = pi_p

    return pi


def main():
    total_num_states = 752 #100 #312020 #50000, 100
    file_df = pd.read_csv('/results/train_state_action_information.csv')
    N = {}
    rho = {}
    S = sorted(set(file_df['s'].tolist() + file_df['sp'].tolist()))
    A = set(file_df['a'].tolist())
    pi_init = dict(zip(list(range(1,total_num_states+1)),[0]*total_num_states))
    gamma = 0.95
    degree = {}
    k_max=500
    action_counts = dict(zip(list(A),[0]*len(A)))

    for index, row in file_df.iterrows():
        if row['s'] in N.keys():
            action_dict = N[row['s']]
            if row['a'] in action_dict.keys():
                N[row['s']][row['a']][row['sp']] = N[row['s']][row['a']].get(row['sp'], 0) + 1
            else:
                N[row['s']][row['a']] = {row['sp']: 1}
            degree[row['s']] = degree.get(row['s'], 0) + 1
            rho[row['s']][row['a']] = rho[row['s']].get(row['a'], 0) + row['r']
        else:
            N[row['s']] = {row['a']: {row['sp']: 1}}
            degree[row['s']] = degree.get(row['s'], 0) + 1
            rho[row['s']] = {row['a']:  row['r']}

    P = MDP(N,rho,S,A,gamma)
    M = PolicyIteration(pi_init, k_max)
    num_iters = 1
    stats_array = np.zeros((num_iters, 1))

    for i in range(num_iters):
        startTime = datetime.now()

        policy = solve_policy_iteration(M, P)
        elapsed_time = datetime.now() - startTime
        stats_array[i, 0] = elapsed_time.total_seconds()
    print("Ave Elapsed time: " + str(np.mean(stats_array)))
    for s,a in policy.items():
        action_counts[a] += 1

    default_action = max(action_counts.items(), key=operator.itemgetter(1))[0]
    with open('/results/policy_iteration.policy', 'w') as f:
        f.write("policy\n")
        for i in range(1, total_num_states+1):
            f.write("%s\n" % policy.get(i,default_action))
    print("Done!")

if __name__ == "__main__":
    main()