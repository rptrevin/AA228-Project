import numpy as np
import pandas as pd
from math import log2
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
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
def kl_divergence(p, q):
    kl_val = 0
    for i in range(len(q)):
        if q[i] > 0:
            if p[i] > 0:
                kl_val += p[i] * log2(p[i]/q[i])
    return kl_val

def act2meds(a):
    pressor = int(np.divide(a + 4, 5))
    fluid   = int((a - 1) % 5 + 1)
    return pressor, fluid



def main():
    traj_dict= {}
    traj_dict['patient_5044'] = []
    traj_dict['patient_5046'] = []
    traj_dict['patient_5200'] = []
    traj_dict['patient_5300'] = []
    mimic_distro_df = pd.read_csv('/home/rptrevin/Documents/AA228/AA228-CS238-Student/project2/results/mimic_distro.policy', header=None)
    eval_distro_df = pd.read_csv('/policy_iteration.policy')
    file_df = pd.read_csv('/train_state_action_information.csv')
    state_vector = [i+1 for i in eval_distro_df.index.tolist()]
    state_action_dict = dict(zip(state_vector, eval_distro_df['policy'].tolist()))
    ex_ai_fluid_counts = [0 for i in range(5)]
    ex_ai_pressor_counts = [0 for i in range(5)]

    ex_clin_fluid_counts = [0 for i in range(5)]
    ex_clin_pressor_counts = [0 for i in range(5)]

    sr_ai_fluid_counts = [0 for i in range(5)]
    sr_ai_pressor_counts = [0 for i in range(5)]

    sr_clin_fluid_counts = [0 for i in range(5)]
    sr_clin_pressor_counts = [0 for i in range(5)]
    poi = {'patient_5044', 'patient_5046', 'patient_5200', 'patient_5300'}
    viable_dataset = set(range(4866, 5366+1))
    viable_dataset = {'patient_'+str(e) for e in viable_dataset}
    for index, row in file_df.iterrows():
        if row['patient'] in viable_dataset:
            ai_a = state_action_dict[row['s']]
            clin_a = row['a']

            ai_tuple = act2meds(ai_a)
            clin_tuple = act2meds(clin_a)

            if row['mortality']==True:

                ex_ai_pressor_counts[ai_tuple[0]-1] += 1
                ex_ai_fluid_counts[ai_tuple[1]-1] += 1

                ex_clin_pressor_counts[clin_tuple[0]-1] += 1
                ex_clin_fluid_counts[clin_tuple[1]-1] += 1
            else:
                sr_ai_pressor_counts[ai_tuple[0]-1] += 1
                sr_ai_fluid_counts[ai_tuple[1]-1] += 1

                sr_clin_pressor_counts[clin_tuple[0]-1] += 1
                sr_clin_fluid_counts[clin_tuple[1]-1] += 1



        if row['patient'] in poi:
            traj_dict[row['patient']].append(state_action_dict[row['s']])


    #distr_deceased = (
        # Physisian
    ex_clin_pressor_counts = [6387.0, 1806.0, 1131.0, 541.0, 3398.0]
    ex_clin_fluid_counts =  [4110.0, 4753.0, 1328.0, 905.0, 2167.0]
        # AI
    ex_ai_pressor_counts = [664.0, 2767.0, 4607.0, 2795.0, 2430.0]
    ex_ai_fluid_counts = [1557.0, 923.0, 2666.0, 4647.0, 3470.0]
    #9: 38
    #distr_survived = (
        # Physisian
    sr_clin_pressor_counts = [11553.0, 3346.0, 1520.0, 700.0, 2787.0]
    sr_clin_fluid_counts = [6841.0, 6620.0, 1350.0, 1404.0, 3691.0]
        # AI
    sr_ai_pressor_counts = [2319.0, 2720.0, 5547.0, 6136.0, 3184.0]
    sr_ai_fluid_counts =  [4709.0, 1595.0, 4085.0, 5473.0, 4044.0]
    cs_sr_fluid = chisquare(sr_ai_fluid_counts, sr_clin_fluid_counts)
    cs_sr_pressor = chisquare(sr_ai_pressor_counts,sr_clin_pressor_counts)
    cs_ex_fluid = chisquare(ex_ai_fluid_counts, ex_clin_fluid_counts)
    cs_ex_pressor = chisquare(ex_ai_pressor_counts, ex_clin_pressor_counts)


    with open('/home/rptrevin/Documents/AA228/AA228-CS238-Student/project2/results/sarsa_stats.txt', 'w') as f:
        ai_pressor_line = 'AI Pressor for expired: '+''.join([str(elem) + "," for elem in ex_ai_pressor_counts])
        f.write(ai_pressor_line[:-1] + "\n")

        ai_fluid_line = 'AI Fluid for expired: '+''.join([str(elem) + "," for elem in ex_ai_fluid_counts])
        f.write(ai_fluid_line[:-1] + "\n")

        clin_pressor_line = 'Physician Pressor for expired: ' + ''.join([str(elem) + "," for elem in ex_clin_pressor_counts])
        f.write(clin_pressor_line[:-1] + "\n")

        clin_fluid_line = 'Physician Fluid for expired: '+''.join([str(elem) + "," for elem in ex_clin_fluid_counts])
        f.write(clin_fluid_line[:-1] + "\n")

        ai_pressor_line = 'AI Pressor for survived: '+''.join([str(elem) + "," for elem in sr_ai_pressor_counts])
        f.write(ai_pressor_line[:-1] + "\n")

        ai_fluid_line = 'AI Fluid for survived: '+''.join([str(elem) + "," for elem in sr_ai_fluid_counts])
        f.write(ai_fluid_line[:-1] + "\n")

        clin_pressor_line = 'Physician Pressor for survived: ' + ''.join([str(elem) + "," for elem in sr_clin_pressor_counts])
        f.write(clin_pressor_line[:-1] + "\n")

        clin_fluid_line = 'Physician Fluid for survived: '+''.join([str(elem) + "," for elem in sr_clin_fluid_counts])
        f.write(clin_fluid_line[:-1] + "\n")

        #Statistical Analysis
        cs_ex_fluid_line = 'Chi Squared Test Expired Fluid : ' + str(cs_ex_fluid) + '\n'
        f.write(cs_ex_fluid_line)

        cs_ex_pressor_line = 'Chi Squared Test Expired Pressor : ' + str(cs_ex_pressor) + '\n'
        f.write(cs_ex_pressor_line)

        cs_sr_fluid_line = 'Chi Squared Test Survived Fluid : ' + str(cs_sr_fluid) + '\n'
        f.write(cs_sr_fluid_line)

        cs_sr_pressor_line = 'Chi Squared Test Survived Pressor : ' + str(cs_sr_pressor) + '\n'
        f.write((cs_sr_pressor_line))

        f.write(clin_fluid_line[:-1] + "\n")
        for key, val in traj_dict.items():
            poi_dict = key + ': ' + ''.join([str(elem) + "," for elem in val])
            f.write(poi_dict[:-1] + "\n")

    f.close()


    print("Done!")
if __name__ == "__main__":
    main()