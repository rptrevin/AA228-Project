import numpy as np
import pandas as pd

class MDP_struct:
    def __init__(self, gamma, A, S, T, R, TR):
        self.gamma = gamma
        self.A = A
        self.S = S
        self.T = T
        self.R = R
        self.TR = TR


def sum_row(row_dict):
    res = 0
    for key, val in row_dict.items():
        res += val
    return res

def MDP(N, rho, S, A, gamma):

    T = N.copy()
    R = rho.copy()
    TR = rho.copy()
    for s in S:
        n_a = 0
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
                    TR[s][a] = 0.0

            else:
                for sp in N[s][a].keys():
                    T[s][a][sp] = N[s][a][sp] / n

                R[s][a] = float(rho[s][a]) / (n)
                TR[s][a] = float(n)

    for s in TR.keys():
        row_sum = sum_row(TR[s])
        for a in TR[s].keys():
            if row_sum > 0:
                TR[s][a] = TR[s][a] / row_sum
            else:
                TR[s][a] = 0

    return MDP_struct(gamma, A, S, T, R, TR)

def main():
    total_num_states = 752 #100 #312020 #50000, 100
    file_df = pd.read_csv('/test_state_action_information.csv')
    N = {}
    rho = {}
    S = [s for s in range(1, total_num_states)] #sorted(set(file_df['s'].tolist() + file_df['sp'].tolist()))
    A = [a for a in range(1, 25)] #set(file_df['a'].tolist())
    gamma = 0.95
    degree = {}

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

    P = MDP(N, rho, S, A, gamma)
    TR = P.TR
    with open('/home/rptrevin/Documents/AA228/AA228-CS238-Student/project2/results/mimic_distro.policy', 'w') as f:
        for i in range(1, total_num_states):
            if i in TR.keys():
                distro = [TR[i].get(a,0) for a in A]
            else:
                distro = [0 for a in A]
            distro_line = ''.join([str(elem)+"," for elem in distro])
            f.write(distro_line[:-1] + "\n")

    print("Done!")
if __name__ == "__main__":
    main()