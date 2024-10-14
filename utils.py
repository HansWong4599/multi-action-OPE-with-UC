"""Plotting and other functions for autsim experiment"""

import numpy as np
import matplotlib.pyplot as plt

def seprate_policies(data, slow_responder=True):
    """Separate the data based on three different policies
    1. with AAC (DTR (BLI, BLI+AAC))
    2. without ACC (DTR (BLI, BLI))
    3. Using AAC from the beginning (DTR (BLI+AAC, ·))

    For more information, please refer to Figure 1 and Table 1 from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876020/

    Parameters
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients
    Returns
    -------
    pi_aac : np.array [None, num_covariates + num_outputs]
        part of data that are assigned to AAC from the beginning
    pi_adaptive : np.array [None, num_covariates + num_outputs]
        part of data that are assigned to policy AAC at second stage
    pi_int : np.array [None, num_covariates + num_outputs]
        part of data that are assigned policy without ACC
    """
    if slow_responder:
        data = data[data[:, 7] == 0, :]
    pi_aac = data[data[:, 8] == -1, :]

    pi_adaptive = data[data[:, 8] == 1, :]
    pi_adaptive = pi_adaptive[pi_adaptive[:, 9] == -1, :]

    pi_int = data[data[:, 8] == 1, :]
    pi_int = pi_int[pi_int[:, 9] == 1, :]

    return pi_aac, pi_int, pi_adaptive

def compute_effect_size(data, slow_responder=True):
    """computes the effect size of the adaptive policy
    as descibed in appendix B
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876020/

    Parameters 
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients

    Returns
    -------
    effect_size : float
        effect size observed in :param data:
    """
    _, pi_int, pi_adaptive = seprate_policies(data, slow_responder=slow_responder)

    AUC11 = (pi_int[:, 10]/2 + pi_int[:, 11] + pi_int[:, 12] + pi_int[:,13]/2) * 12
    AUC1n1 = (pi_adaptive[:, 10]/2 + pi_adaptive[:, 11] + pi_adaptive[:, 12] + pi_adaptive[:,13]/2) * 12
    
    m11 = np.mean(AUC11); s11=np.std(AUC11)
    m1n1 = np.mean(AUC1n1); s1n1=np.std(AUC1n1)
    return (m1n1 - m11)/np.sqrt((s11**2 + s1n1**2)/2)

def compute_observational_policy_value(data, slow_responder=True):
    """computes observational estimate of the policy value
    Parameters 
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients

    Returns
    -------
    pi_acc_v : float
        value of AAC policy from the beginning
    pi_int_v : float
        value of policy without ACC
    pi_adaptive_v : float
        value of policy that are assigned policy AAC at second stage
    """ 
    pi_aac, pi_int, pi_adaptive = seprate_policies(data, slow_responder=slow_responder)

    pi_acc_v =np.mean(pi_aac[:,10:14], axis=0)
    pi_int_v = np.mean(pi_int[:,10:14], axis=0)
    pi_adaptive_v = np.mean(pi_adaptive[:,10:14], axis=0)

    return pi_acc_v, pi_int_v, pi_adaptive_v

def plot_autism(data_list, evaluations=None, title='', fontsize=30,
                      markersize=18, linewidth=3, fontscale=0.6):
    """generates plot for autism example
    Parameters
    ----------
    data_list : float np.array num_all_methods x num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    evaluations : dictionary
        key : legend of the observation, eg. 'BLI+AAC'
        value : (int, float, color, marker)
            tuple of week, value, point color, and marker type
    confounding : float
        amount of confounding in data generation process
    """

    weeks = [0, 12, 24, 36]
    labels = ['Truth', 'DM','DM_IVs','DM_Aux','DM_Pxy']
    fmts = ['-o','--v','--p','--*',':D']

    # plot estimated mean trajectories under three target policies.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    for i, data in enumerate(data_list):
        pi_aac, pi_int, pi_adaptive = compute_observational_policy_value(data, slow_responder=True)

        # target policy 1
        ax1.plot(weeks, pi_aac, fmts[i],
                 markersize=markersize, label= labels[i], linewidth=linewidth)

        # target policy 2
        ax2.plot(weeks, pi_int, fmts[i],
                 markersize=markersize, label=labels[i], linewidth=linewidth)

        # target policy 3
        ax3.plot(weeks, pi_adaptive, fmts[i],
                 markersize=markersize, label=labels[i], linewidth=linewidth)


    ax1.set_title('Policy I (BLI+AAC, ·)', fontsize=fontsize)
    ax1.set_ylabel("Avg. Speech Utterances", fontsize=fontsize * fontscale)
    # ax2.set_title('Policy II (BLI, INT)', fontsize=fontsize)
    # ax1.set_ylim(29,41)
    # ax2.set_ylim(26,31)
    # ax3.set_ylim(29,39)

    ax3.set_title('Policy II (BLI, BLI+AAC)', fontsize=fontsize)


    #
    for ax in [ax1, ax3]:
        ax.set_xlabel("Weeks", fontsize=fontsize * fontscale)
        ax.tick_params(axis='both', labelsize=fontsize * fontscale)
        ax.grid(axis='y', linestyle='--', color='gray' , linewidth=2)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)


    # Plot evaluated values
    if evaluations is not None:
        for key in evaluations.keys():
            week, value, color, marker = evaluations[key]
            plt.plot(week, value, marker, color=color, 
                    markersize=markersize, label=key, linewidth=linewidth, 
                    fillstyle='full')

    plt.legend(loc='best', fontsize=fontsize * fontscale)
    # plt.savefig("output3.jpg", bbox_inches='tight')
    # plt.tight_layout()
    plt.show()



def plot_autism_design_sensitivity(data, title='', fontsize=30, 
                      markersize=23, linewidth=6, fontscale=0.8):
    """genrates plot for autism example, design sensitivity 
    Parameters
    ----------
    data : dictionary with four (key, value)
        - 'our' : float, np.array : computed lower bounds with our method
        - 'naive' : float, np.array : computed lower bounds with naive bounds
        - 'Gammas' : float, np.array
        - 'aac' : float : value of aac policy
        - 'adaptive' : float : value of adaptive policy
    """
    length = len(data['Gammas'])
    aac = np.array([data['aac']] * length)
    adaptive = np.array([data['adaptive']] * length)

    plt.plot(data['Gammas'], aac, '-o', color = 'black', 
                markersize=markersize, label='AAC (True)', linewidth=linewidth)
    plt.plot(data['Gammas'], adaptive, '-*', color = 'black', 
                markersize=markersize/fontscale, label='BLI + AAC (True)', linewidth=linewidth)
    plt.plot(data['Gammas'], data['our'], '-.X', color = 'black', 
                markersize=markersize, label='BLI + AAC (Ours)', linewidth=linewidth)
    plt.plot(data['Gammas'], data['naive'], '-.P', color = 'black', 
                markersize=markersize, label='BLI + AAC (Naive)', linewidth=linewidth)
    
    ymax = data['adaptive'] + 2.0
    ymin = np.min(data['our']) - 10.0

    plt.vlines([data['cross'][0]] , ymax, ymin, 
            linestyles='--', colors='gray', linewidth=linewidth)
    plt.vlines([data['cross'][1]] , ymax, ymin,  
            linestyles='--', colors='gray', linewidth=linewidth)

    plt.title(title, 
                fontsize=fontsize, y=1.02)
    plt.legend(loc=4, fontsize=fontsize*fontscale)
    plt.xlabel("Level of confounding ($\Gamma$)", fontsize=fontsize)
    plt.ylabel(r"outcome $\mathbb{E}[Y(\bar A_{1:T})]$", fontsize=fontsize)
    plt.xticks(data['Gammas'].tolist() + [data['cross'][0], data['cross'][1]], 
            fontsize=fontsize*fontscale)
    plt.yticks(fontsize=fontsize*fontscale)
    plt.ylim([ymin, ymax])
    plt.grid(linewidth=3)