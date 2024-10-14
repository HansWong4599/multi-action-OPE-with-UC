import numpy as np
from numpy.random import multivariate_normal as mvrnorm

# parameter setting
# 6 treatment, 6 instruments
# 1 unmeasured confounder
# 1 confounder proxy (W)

# alpha is coefficient of U on W,A,R and S prime.
alpha_A = np.array([
    [1, 1, 1, 1, 1, 1]
])
alpha_R = np.array([2.5])
alpha_S_prime = np.array([5])
alpha_W = np.array([2])

# lamb is coefficient of S on A, R and S prime.

lamb_A = np.array([[1] * 6])
# lamb_A = np.array([[0.15682667, 0.02345362, 0.04693393, 0.25830943, 0.25627336, 0.25820299]])
lamb_R = np.array([1])
lamb_S_prime = np.array([0.1])

# the coe of Z(IVs) on A
eta = np.diag([1] * 6)

# the coe of A(treatment) on R/S_prime(outcome)
beta_R = np.array([1] * 6)
beta_S_prime = np.array([0.1] * 6)

# the dim of variables
dimZ = eta.shape[0]
dimS = lamb_A.shape[0]
dimA = alpha_A.shape[1]
dimU = alpha_A.shape[0]
dimW = alpha_A.shape[0]
dimR = 1


para = {
    'alpha_A': alpha_A,
    'alpha_R': alpha_R,
    'alpha_S_prime': alpha_S_prime,
    'alpha_W': alpha_W,
    'lamb_A': lamb_A,
    'lamb_R': lamb_R,
    'lamb_S_prime': lamb_S_prime,
    'eta': eta,
    'beta_R': beta_R,
    'beta_S_prime': beta_S_prime,
    'dimZ': dimZ,
    'dimA': dimA,
    'dimU': dimU
}

# define some functions to generate the simulation data

def gene_S0(n_samples):
    S0 = np.random.normal(size=(n_samples, 1), loc=0, scale=1)
    return S0
def tar_Pi(n_samples):  # the target policy we want to evaluate
    return np.full((n_samples, dimA),1)

def gene_UC(n_samples, para, St, is_tar_Pi = False):

    # One stage data generating in the presence of unobserved confounders(UC)

    # parameter setting
    alpha_A = para['alpha_A']
    alpha_R = para['alpha_R']
    alpha_S_prime = para['alpha_S_prime']
    alpha_W = para['alpha_W']

    lamb_A = para['lamb_A']
    lamb_R = para['lamb_R']
    lamb_S_prime = para['lamb_S_prime']

    eta = para['eta']
    beta_R = para['beta_R']
    beta_S_prime = para['beta_S_prime']

    dimZ = para['dimZ']
    dimU = para['dimU']

    Ut = mvrnorm(mean=np.zeros(dimU), cov=np.eye(dimU), size=n_samples)
    Wt = np.dot(Ut, alpha_W) + np.random.normal(loc=0, scale=1, size=n_samples)
    # Ut = np.zeros((n_samples, dimU))
    Zt = mvrnorm(mean=np.zeros(dimZ), cov=np.eye(dimZ), size=n_samples)
    if is_tar_Pi: # do(At)
        At = tar_Pi(n_samples)
    else:
        At = np.dot(Ut, alpha_A) + np.dot(Zt, eta) + np.dot(St, lamb_A) + mvrnorm(mean=np.zeros(dimA), cov=np.eye(dimA), size=n_samples)
    Rt = np.dot(Ut, alpha_R) + np.dot(At, beta_R) + np.dot(St, lamb_R) + np.random.normal(loc=0, scale=1, size=n_samples)
    S_prime = np.dot(Ut, alpha_S_prime) + np.dot(At, beta_S_prime) + np.dot(St, lamb_S_prime) + np.random.normal(loc=0, scale=1, size=n_samples)

    return Wt.reshape(-1,1), Zt, At, Rt.reshape(-1,1), S_prime.reshape(-1,1)

def gene_NUC(n_samples, para, St):

    # One stage data generating without unobserved confounders(NUC)

    # parameter setting
    # no confounders

    lamb_R = para['lamb_R']
    lamb_S_prime = para['lamb_S_prime']
    beta_R = para['beta_R']
    beta_S_prime = para['beta_S_prime']

    At = tar_Pi(n_samples)
    Rt = np.dot(At, beta_R) + np.dot(St, lamb_R) + np.random.normal(loc=0, scale=1, size=n_samples)
    S_prime = np.dot(At, beta_S_prime) + np.dot(St, lamb_S_prime) + np.random.normal(loc=0, scale=1, size=n_samples)

    return At, Rt.reshape(-1,1), S_prime.reshape(-1,1)

def gene_MDP(n_samples, n_IVs):
    # generate simulation dataset 'MDP' for evaluation

    # n_samples = 100  # the number of samples/trajectories
    T = 50  # the number of stages
    # n_IVs = 6 # the number of IVs we use

    S0 = gene_S0(n_samples)
    MDP = np.copy(S0)
    St = S0

    # If DM-IV, all IVs, if auxiliary variables, any of the columns
    selected_Zt_idx = np.arange(dimZ) if n_IVs == dimZ else np.random.choice(dimZ, n_IVs, replace=False)

    for t in range(T):
        Wt, Zt, At, Rt, S_prime = gene_UC(n_samples, para, St)
        MDP = np.hstack((MDP, Wt, Zt[:,selected_Zt_idx], At, Rt, S_prime))
        St = S_prime

    return {'data': MDP, 'stage': T, 'dimW':dimW, 'dimS':dimS, 'dimZ': n_IVs, 'dimA':dimA, 'dimR':dimR}

def gene_GroundTruth():
    # Approximate the true value under tar_Pi by MC method (ground truth).

    n_samples = 1000  # the number of samples/trajectories
    n_sims = 100  # the number of replications of simulations
    T = 50  # the number of stages
    gamma = 0.9
    esti = 0

    for i in range(n_sims):

        S0 = gene_S0(n_samples)
        MDP_Pi = np.copy(S0)
        St = S0

        # Actually, the true value after do(A) operation is consistent regardless of the presence or absence of Ut.
        # This is because the do operation cuts off the edge from Ut to At,
        # i.e., there is no direct causal effect from Ut to At.

        for t in range(T):
            At, Rt, S_prime = gene_NUC(n_samples, para, St)
            # Zt, At, Rt, S_prime = gene_UC(n_samples, para, St, is_tar_Pi=True)
            MDP_Pi = np.hstack((MDP_Pi, At, Rt, S_prime))
            St = S_prime

            esti = esti + sum(Rt) / n_samples  # average reward
            # esti = esti + (gamma ** t) * sum(Rt) / n_samples # discounted reward

    esti = esti / n_sims / T  # remove T if discounted reward

    return esti



