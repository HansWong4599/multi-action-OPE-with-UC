import numpy as np
from numpy.linalg import inv

from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

from data.simulated_data_generation import gene_MDP, tar_Pi, gene_GroundTruth, gene_S0


def OLS(MDP):
    # vectorize the data for further use
    data = MDP['data']
    n_samples = data.shape[0]  # the number of trajectories
    T = MDP['stage']  # the number of stages
    dimA = MDP['dimA']
    dimS = MDP['dimS']
    dimZ = MDP['dimZ']
    dimR = MDP['dimR']
    dimW = MDP['dimW']

    # Define Q function and Value function
    # Q function: a linear function of (St,At).
    def Q(beta, n_samples, St, At):
        return np.dot(np.concatenate((St, At), axis=1), beta)

    # Value function(V)
    def V(St, n_samples, tar_Pi, beta):  # Treat original V_pi(St)
        # deterministic policy Pi: here we just test Pi=1. The whole function can be generalized to random policy later.
        V_St = Q(beta, n_samples, St, tar_Pi(n_samples))
        return V_St

    part1_Q = 0
    part2_Q = 0
    temp = 0
    Sprime_esti = data[:, 0:  dimS]
    for t in range(T):
        St = data[:, (dimS + dimW + dimZ + dimA + dimR) * t: (dimS + dimW + dimZ + dimA + dimR) * t + dimS]
        At = data[:,
             (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ: (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ + dimA]
        Rt = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) - dimR]
        S_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1): (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS]

        Linear_model_R = LinearRegression().fit(np.hstack((At, St)), Rt)
        Linear_model_Sprime = LinearRegression().fit(np.hstack((At, St)), S_prime.flatten())

        hbeta_R = Linear_model_R.coef_[:dimA]
        hbeta_Sprime = Linear_model_Sprime.coef_[:dimA]
        hlamb_R = Linear_model_R.coef_[dimA:dimA+dimS]
        hlamb_Sprime = Linear_model_Sprime.coef_[dimA:dimA+dimS]

        St = Sprime_esti.reshape(-1, 1)
        R_esti = np.dot(tar_Pi(n_samples), hbeta_R) + np.dot(St, hlamb_R)
        Sprime_esti = np.dot(tar_Pi(n_samples), hbeta_Sprime) + np.dot(St, hlamb_Sprime)
        temp = temp + np.sum(R_esti) / n_samples

        xi_t_Q = np.concatenate((St, At), axis=1)
        xi_prime_Q = np.concatenate((Sprime_esti.reshape(-1, 1), tar_Pi(n_samples)), axis=1)
        part1_Q = part1_Q + np.matmul(xi_t_Q.T, (xi_t_Q - xi_prime_Q))
        part2_Q = part2_Q + np.sum(xi_t_Q * R_esti[:, np.newaxis], axis=0)

    beta_Q_NUC = np.linalg.solve(part1_Q, part2_Q)

    # DM estimator:
    S0_test = gene_S0(n_samples)
    esti = V(S0_test, n_samples, tar_Pi, beta_Q_NUC)
    esti = np.sum(esti) / n_samples / T

    print("DM estimator with OLS: ", temp/T,"Q learning with OLS",esti)
    return temp/T


def IV(MDP):
    # vectorize the data for further use
    data = MDP['data']
    n_samples = data.shape[0]  # the number of trajectories
    T = MDP['stage']  # the number of stages
    dimA = MDP['dimA']
    dimS = MDP['dimS']
    dimZ = MDP['dimZ']
    dimR = MDP['dimR']
    dimW = MDP['dimW']

    # Calculating the range of steps [S*dimS,Z*dimZ,A*dimA,R*dimR,Sprime]
    step_size = dimS + dimW + dimZ + dimA + dimR
    S_start = 0
    Z_start = dimS + dimW
    A_start = dimS + dimW + dimZ
    R_start = dimS + dimW + dimZ + dimA

    S_indices = np.array([i * step_size + S_start + j for i in range(T) for j in range(dimS)])
    Z_indices = np.array([i * step_size + Z_start + j for i in range(T) for j in range(dimZ)])
    A_indices = np.array([i * step_size + A_start + j for i in range(T) for j in range(dimA)])
    R_indices = np.array([i * step_size + R_start + j for i in range(T) for j in range(dimR)])
    Sprime_indices = np.array([(i + 1) * step_size + S_start + j for i in range(T) for j in range(dimS)])

    Svec = data[:, S_indices].reshape(-1, dimS)
    Zvec = data[:, Z_indices].reshape(-1, dimZ)
    Avec = data[:, A_indices].reshape(-1, dimA)
    Rvec = data[:, R_indices].reshape(-1)
    Sprimevec = data[:, Sprime_indices].reshape(-1, dimS)



    # Define Q function and Value function
    # Q function: a linear function of (St,At).
    def Q(beta, n_samples, St, At):
        return np.dot(np.concatenate((St, At), axis=1), beta)

    # Value function(V)
    def V(St, n_samples, tar_Pi, beta):  # Treat original V_pi(St)
        # deterministic policy Pi: here we just test Pi=1. The whole function can be generalized to random policy later.
        V_St = Q(beta, n_samples, St, tar_Pi(n_samples))
        return V_St

    part1_Q = 0
    part2_Q = 0
    St_esti = data[:, 0:  dimS]
    for t in range(T):
        St = data[:, (dimS + dimW + dimZ + dimA + dimR) * t: (dimS + dimW + dimZ + dimA + dimR) * t + dimS]
        Zt = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW : (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ]
        At = data[:,
             (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ: (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ + dimA]
        Rt = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) - dimR]
        S_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1): (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS]

        Linear_model_A = LinearRegression().fit(np.hstack((Zt, St)), At)
        hatA = Linear_model_A.predict(np.hstack((Zt, St)))

        hbeta_R = LinearRegression().fit(np.hstack((hatA, St)), Rt).coef_[:dimA]
        hbeta_Sprime = LinearRegression().fit(np.hstack((hatA, St)), S_prime.flatten()).coef_[:dimA]
        hlamb_R = LinearRegression().fit(np.hstack((hatA, St)), Rt).coef_[dimA:dimA + dimS]
        hlamb_Sprime = LinearRegression().fit(np.hstack((hatA, St)), S_prime.flatten()).coef_[dimA:dimA + dimS]


        R_esti = np.dot(At, hbeta_R) + np.dot(St_esti.reshape(-1,1), hlamb_R)
        Sprime_esti = np.dot(At, hbeta_Sprime) + np.dot(St_esti.reshape(-1,1), hlamb_Sprime)

        xi_t_Q = np.concatenate((St, At), axis=1)
        xi_prime_Q = np.concatenate((Sprime_esti.reshape(-1, 1), tar_Pi(n_samples)), axis=1)
        part1_Q = part1_Q + np.matmul(xi_t_Q.T, (xi_t_Q - xi_prime_Q))
        part2_Q = part2_Q + np.sum(xi_t_Q * R_esti[:, np.newaxis], axis=0)
        St_esti = Sprime_esti

    beta_Q_NUC = np.linalg.solve(part1_Q, part2_Q)

    # DM estimator:
    S0_test = gene_S0(n_samples)
    esti = V(S0_test, n_samples, tar_Pi, beta_Q_NUC)
    esti = np.sum(esti) / n_samples

    print("Q learning with IVs: ",esti)
    return esti


def proxy(MDP):
    # prepare the data for further use
    data = MDP['data']
    dimA = MDP['dimA']
    dimS = MDP['dimS']
    dimZ = MDP['dimZ']
    dimR = MDP['dimR']
    dimW = MDP['dimW']

    gamma = 0.99
    n_samples = data.shape[0]  # the number of trajectories
    T = MDP['stage']  # the number of stages

    # Q function: a linear function of (St,At).
    def Q(beta, n_samples, St, At):
        return np.dot(np.concatenate((St, At), axis=1), beta)

    # Value function(V)
    def V(St, n_samples, tar_Pi, beta):  # Treat original V_pi(St)
        # deterministic policy Pi: here we just test Pi=1. The whole function can be generalized to random policy later.
        V_St = Q(beta, n_samples, St, tar_Pi(n_samples))
        return V_St

    # estimate the parameters beta in both Q function and Value function
    part1_Q = 0
    part2_Q = 0
    St_esti = data[:, 0:  dimS]
    for t in range(T):
        St = data[:, (dimS + dimW + dimZ + dimA + dimR) * t: (dimS + dimW + dimZ + dimA + dimR) * t + dimS]
        Wt = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS : (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW]
        Zt = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW : (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ]
        At = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ: (
                                                                                              dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ + dimA]
        Rt = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) - dimR]
        S_prime = data[:,
                  (dimS + dimW + dimZ + dimA + dimR) * (t + 1): (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS]
        A_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS + dimW + dimZ: (
                                                                                                         dimS + dimW + dimZ + dimA + dimR) * (
                                                                                                         t + 1) + dimS + dimW + dimZ + dimA]

        Linear_model_W = LinearRegression().fit(np.hstack((Zt, St, At)), Wt)
        hatW = Linear_model_W.predict(np.hstack((Zt, St, At)))

        hbeta_R = LinearRegression().fit(np.hstack((At, St, hatW)), Rt).coef_[:dimA]
        hbeta_Sprime = LinearRegression().fit(np.hstack((At, St, hatW)), S_prime.flatten()).coef_[:dimA]
        hlamb_R = LinearRegression().fit(np.hstack((At, St, hatW)), Rt).coef_[dimA:dimA + dimS]
        hlamb_Sprime = LinearRegression().fit(np.hstack((At, St, hatW)), S_prime.flatten()).coef_[dimA:dimA + dimS]

        R_esti = np.dot(At, hbeta_R) + np.dot(St_esti.reshape(-1, 1), hlamb_R)
        Sprime_esti = np.dot(At, hbeta_Sprime) + np.dot(St_esti.reshape(-1, 1), hlamb_Sprime)

        xi_t_Q = np.concatenate((St, At), axis=1)
        xi_prime_Q = np.concatenate((Sprime_esti.reshape(-1, 1), tar_Pi(n_samples)), axis=1)
        part1_Q = part1_Q + np.matmul(xi_t_Q.T, (xi_t_Q - xi_prime_Q))
        part2_Q = part2_Q + np.sum(xi_t_Q * R_esti[:, np.newaxis], axis=0)
        St_esti = Sprime_esti

    beta_Q_NUC = np.linalg.solve(part1_Q, part2_Q)

    # DM estimator:
    S0_test = gene_S0(n_samples)
    esti = V(S0_test, n_samples, tar_Pi, beta_Q_NUC)
    esti = np.sum(esti) / n_samples

    print("Q learning with proxy: ", esti)
    return esti

def multi_treatment_deconfound(MDP, nfact = 1):
    # =============================================================================
    # IV+MDP. Off policy evaluation: estimate the averaged treatment effect in infinite-horizon MDP settings
    # -------------------------------------------------------------------------
    #  parameters:
    #  MDP: dataframe, which contains all observed data formalized
    #       as (St,Zt,At,Rt) from stage 1 to T.
    #  nfact: the number of factor model (Best to be the same as the number of unobserved confounders.)
    #  Pi: the target policy we want to evaluate
    #  domain_At: the domain/support of action space
    #  domain_Zt: the domain/support of IV space
    #  max_iter: maximum number of iterations in fitted-Q evaluation
    #  epsilon: error bound in fitted-Q evaluation
    #
    #
    # =============================================================================

    # vectorize the data for further use
    data = MDP['data']
    n_samples = data.shape[0] # the number of trajectories
    T = MDP['stage'] # the number of stages
    dimA = MDP['dimA']
    dimS = MDP['dimS']
    dimZ = MDP['dimZ']
    dimR = MDP['dimR']
    dimW = MDP['dimW']

    gamma = 0.99


    #Define Q function and Value function
    # Q function: a linear function of (St,At).
    def Q(beta, n_samples, St, At):
        return np.dot(np.concatenate((St, At), axis=1), beta)

    # Value function(V)
    def V(St, n_samples, tar_Pi, beta):  # Treat original V_pi(St)
        # deterministic policy Pi: here we just test Pi=1. The whole function can be generalized to random policy later.
        V_St = Q(beta, n_samples, St, tar_Pi(n_samples))
        return V_St

    part1_Q = 0
    part2_Q = 0
    St_esti = data[:, 0:  dimS]
    for t in range(T):
        St = data[:, (dimS + dimW + dimZ + dimA + dimR) * t: (dimS + dimW + dimZ + dimA + dimR) * t + dimS]
        Zt = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW : (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ]
        At = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ: (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ + dimA]
        Rt = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) - dimR]
        S_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1): (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS]

        ### estimate models
        # (1) regress action(A) on IV(Z) and covarites(S)
        #    regress reward(R) and next state (Sprime) on IV(Z), action(A) and covarites(S)
        linear_model_A = LinearRegression().fit(np.hstack((Zt, St)), At)
        Linear_model_R = LinearRegression().fit(np.hstack((At, Zt, St)), Rt)
        Linear_model_Sprime = LinearRegression().fit(np.hstack((At, Zt, St)), S_prime.flatten())
        heta = linear_model_A.coef_
        hxi_R = Linear_model_R.coef_
        hxi_Sprime = Linear_model_Sprime.coef_


        # (2) Factor analysis
        A_res = linear_model_A.predict(np.hstack((Zt, St))) - At
        fa = FactorAnalyzer(rotation=None, n_factors=nfact)
        fa.fit(A_res)
        efa_fit = fa.loadings_

        # (3) Calculate halpha_A
        var_A_res = np.var(A_res, axis=0, ddof=1)
        halpha_A = efa_fit * np.sqrt(var_A_res).reshape(-1, 1)

        # (4) Calculate hgamma
        cov_matrix = np.cov(A_res, rowvar=False)
        hgamma = inv(cov_matrix) @ halpha_A

        # (5) Estimate confounding bias (coefs of unobserved confounders)
        halpha_R = -inv(hgamma.T @ heta @ heta.T @ hgamma) @ (hgamma.T @ heta) @ hxi_R[dimA : dimA + dimZ + dimS]
        halpha_Sprime = -inv(hgamma.T @ heta @ heta.T @ hgamma) @ (hgamma.T @ heta) @ hxi_Sprime[dimA : dimA + dimZ+ dimS]

        # (6) Estimate effects (the coefs of At on Rt and Sprime)
        hbeta_R = hxi_R[:dimA] - hgamma @ halpha_R
        hbeta_Sprime = hxi_Sprime[:dimA] - hgamma @ halpha_Sprime

        hlamb_R = hxi_R[dimA + dimZ: dimA + dimZ + dimS] + hgamma @ halpha_R @ heta[:, dimZ: dimZ + dimS]
        hlamb_Sprime = hxi_Sprime[dimA + dimZ: dimA + dimZ + dimS] + hgamma @ halpha_Sprime @ heta[:, dimZ: dimZ + dimS]

        # (7) estimate the parameters beta in both Q function and Value function

        # Extract S and A at current timestep as the feature, where np.ones denote intercept

        R_esti = np.dot(At, hbeta_R) + np.dot(St_esti.reshape(-1,1), hlamb_R)
        Sprime_esti = np.dot(At, hbeta_Sprime) + np.dot(St_esti.reshape(-1,1), hlamb_Sprime)

        xi_t_Q = np.concatenate((St, At), axis=1)
        xi_prime_Q = np.concatenate((Sprime_esti.reshape(-1, 1), tar_Pi(n_samples)), axis=1)
        part1_Q = part1_Q + np.matmul(xi_t_Q.T, (xi_t_Q - xi_prime_Q))
        part2_Q = part2_Q + np.sum(xi_t_Q * R_esti[:, np.newaxis], axis=0)
        St_esti = Sprime_esti

    beta_Q_NUC = np.linalg.solve(part1_Q, part2_Q)

    # DM estimator:
    S0_test = gene_S0(n_samples)
    esti = V(S0_test, n_samples, tar_Pi, beta_Q_NUC)
    esti = np.sum(esti) / n_samples

    print("Q learning with aux", esti)
    return esti


def DM_Q(MDP):
    # =============================================================================
    # Direct method (standard linear Q-learning)
    # -------------------------------------------------------------------------
    #  parameters:
    #  MDP: dataframe, which contains all observed data formalized
    #       as (St,At(multi-action),Rt) from stage 1 to T.
    #  dim*: the dimension of action/State/Reward
    #  n_samples: the number of trajectories
    #  T: the number of stages
    #
    #
    # =============================================================================

    # prepare the data for further use
    data = MDP['data']
    dimA = MDP['dimA']
    dimS = MDP['dimS']
    dimZ = MDP['dimZ']
    dimR = MDP['dimR']
    dimW = MDP['dimW']

    gamma = 0.99
    n_samples = data.shape[0]  # the number of trajectories
    T = MDP['stage'] # the number of stages

    # Q function: a linear function of (St,At).
    def Q(beta, n_samples, St, At):
        return np.dot(np.concatenate((St, At), axis=1), beta)

    # Value function(V)
    def V(St, n_samples, tar_Pi, beta):  # Treat original V_pi(St)
        # deterministic policy Pi: here we just test Pi=1. The whole function can be generalized to random policy later.
        V_St = Q(beta, n_samples, St, tar_Pi(n_samples))
        return V_St

    # estimate the parameters beta in both Q function and Value function
    part1_Q = 0
    part2_Q = 0
    for t in range(T):
        St = data[:, (dimS + dimW + dimZ + dimA + dimR) * t: (dimS + dimW + dimZ + dimA + dimR) * t + dimS]
        At = data[:, (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ: (dimS + dimW + dimZ + dimA + dimR) * t + dimS + dimW + dimZ + dimA]
        Rt = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) - dimR]
        S_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1): (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS]
        A_prime = data[:, (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS + dimW + dimZ: (dimS + dimW + dimZ + dimA + dimR) * (t + 1) + dimS + dimW + dimZ + dimA]

        # Extract the feature (S,A) at current timestep
        xi_t_Q = np.concatenate((St, At), axis=1)

        # Construct the feature (S_prime,do(A)) at next timestep
        xi_prime_Q = np.concatenate((S_prime, tar_Pi(n_samples)), axis=1)

        part1_Q = part1_Q + np.matmul(xi_t_Q.T, (xi_t_Q - xi_prime_Q))
        part2_Q = part2_Q + np.sum(xi_t_Q * Rt[:, np.newaxis], axis=0)

    beta_Q_NUC = np.linalg.solve(part1_Q, part2_Q)

    # Generates the initial state (S0_test) of the test data.
    S0_test = gene_S0(n_samples)
    esti = V(S0_test, n_samples, tar_Pi, beta_Q_NUC)
    esti = np.sum(esti) / n_samples

    print("DM estimator (standard linear Q-learning): ", esti)
    return esti


if __name__ == '__main__':
    np.random.seed(1)

    # MDP_NUC, ground_truth = gene_MDP(is_confounded=False)
    # DM_Q(MDP_NUC)

    n_sims = 100 # Number of repetitions per method

    esti_IVs = np.zeros((n_sims, 10))
    esti_one_IV = np.zeros((n_sims, 10))
    esti_Aux = np.zeros((n_sims, 10))
    esti_OLS = np.zeros((n_sims, 10))
    esti_Q = np.zeros((n_sims, 10))
    esti_proxy = np.zeros((n_sims, 10))

    # Number of samples per experiment
    # [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    n_samples_all = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])

    for rep in range(n_sims):
        print("rep=", rep, ":\n")

        for i in range(len(n_samples_all)):
            n_samples = n_samples_all[i]
            MDP_UC_oneIV = gene_MDP(n_samples, 1) # Returns only an IV
            MDP_UC = gene_MDP(n_samples, 6)

            esti_Q[rep, i] = DM_Q(MDP_UC)
            esti_one_IV[rep, i] = IV(MDP_UC_oneIV)
            # esti_OLS[rep, i] = OLS(MDP_UC)
            esti_IVs[rep, i] = IV(MDP_UC)
            esti_Aux[rep, i] = multi_treatment_deconfound(MDP_UC_oneIV)
            esti_proxy[rep, i] = proxy(MDP_UC_oneIV)

    groundtruth = gene_GroundTruth()
    print("True value under target policy:", groundtruth)


    MSE_Q = np.sum((esti_Q-groundtruth)**2,0)/n_sims
    MSE_IV = np.sum((esti_IVs - groundtruth) ** 2, 0) / n_sims
    MSE_Aux = np.sum((esti_Aux - groundtruth) ** 2, 0) / n_sims
    MSE_oneIV = np.sum((esti_one_IV-groundtruth)**2,0)/n_sims
    MSE_proxy = np.sum((esti_proxy-groundtruth)**2,0)/n_sims

    Bias_Q = np.sum(esti_Q, 0) / n_sims - groundtruth
    Bias_IV = np.sum(esti_IVs, 0) / n_sims - groundtruth
    Bias_Aux = np.sum(esti_Aux, 0) / n_sims - groundtruth
    Bias_oneIV = np.sum(esti_one_IV, 0) / n_sims - groundtruth
    Bias_proxy = np.sum(esti_proxy, 0) / n_sims - groundtruth

    trajectory = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # plot
    # 创建大图，包含两组断裂轴图
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])

    # 组 1：图 1 和 图 2
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # 组 2：图 3 和 图 4
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)

    # 自定义颜色和透明度
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 自定义颜色
    markers = ['x', 'o', 's', '^', 'D']  # 自定义标记样式
    linestyles = ['--', '-', '-.', ':', '-']  # 不同的线条样式

    #左上的图画oneIV，异常值
    ax1.plot(trajectory, np.log(MSE_oneIV)[0:len(trajectory)], marker=markers[0], color=colors[0], linestyle=linestyles[0], label='DM_oneIV')
    ax1.grid(axis='y', linestyle='--')
    ax1.set_title('log (Relative MSE)')
    ax1.tick_params(labelbottom=False)
    ax1.spines['bottom'].set_visible(False)  # 隐藏底边框

    #左下的图画其余方法，正常值
    ax2.plot(trajectory, np.log(MSE_Q)[0:len(trajectory)], marker=markers[1], color=colors[1], linestyle=linestyles[1], label='DM')
    ax2.plot(trajectory, np.log(MSE_IV)[0:len(trajectory)], marker=markers[2], color=colors[2], linestyle=linestyles[2], label='DM_IVs')
    ax2.plot(trajectory, np.log(MSE_Aux)[0:len(trajectory)], marker=markers[3], color=colors[3], linestyle=linestyles[3], label='DM_Aux')
    ax2.plot(trajectory, np.log(MSE_proxy)[0:len(trajectory)], marker=markers[4], color=colors[4], linestyle=linestyles[4], label='DM_Pxy')
    ax2.set_ylim(top = -0.5)

    ax2.grid(axis='y', linestyle='--')
    ax2.set_xlabel("Number of trajectories")
    ax2.spines['top'].set_visible(False)  # 隐藏底边框

    # 右上的图画oneIV，异常值
    ax3.plot(trajectory, np.log(abs(Bias_oneIV))[0:len(trajectory)], marker=markers[0], color=colors[0], linestyle=linestyles[0], label='DM_oneIV')
    ax3.set_title('log (Relative Absolute Bias)')
    ax3.grid(axis='y', linestyle='--')
    ax3.tick_params(labelbottom=False)
    ax3.spines['bottom'].set_visible(False)  # 隐藏底边框

    # 右下的图画其余测量值，正常
    ax4.plot(trajectory, np.log(abs(Bias_Q))[0:len(trajectory)], marker=markers[1], color=colors[1], linestyle=linestyles[1], label='DM')
    ax4.plot(trajectory, np.log(abs(Bias_IV))[0:len(trajectory)], marker=markers[2], color=colors[2], linestyle=linestyles[2], label='DM_IVs')
    ax4.plot(trajectory, np.log(abs(Bias_Aux))[0:len(trajectory)], marker=markers[3], color=colors[3], linestyle=linestyles[3], label='DM_Aux')
    ax4.plot(trajectory, np.log(abs(Bias_proxy))[0:len(trajectory)], marker=markers[4], color=colors[4], linestyle=linestyles[4], label='DM_Pxy')
    ax4.set_xlabel("Number of trajectories")
    ax4.grid(axis='y', linestyle='--')
    ax4.spines['top'].set_visible(False)  # 隐藏底边框



    # 在右边两个图中，将图例放在一起
    lines_labels = [ax3.get_legend_handles_labels(), ax4.get_legend_handles_labels()]

    # lines_labels[0][0] 是 ax3 中的 line 对象，lines_labels[1][0] 是 ax4 的
    lines = lines_labels[0][0] + lines_labels[1][0]  # 合并 ax3 和 ax4 的 line 对象
    labels = lines_labels[0][1] + lines_labels[1][1]  # 合并 ax3 和 ax4 的标签

    # 将合并后的图例放置在合适的位置，比如右下角或其他位置
    ax4.legend(lines, labels, loc="upper right", ncol=1)

    # 为所有图添加锯齿效果
    d = .025
    kwargs = dict(transform=ax1.transAxes, color='red', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d, +d), (-d, +d), **kwargs)
    ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax4.transAxes)
    ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


    # plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

