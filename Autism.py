
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.numbers import Pi

from data.autism.utils import *
from data.autism.data_gen import DataGen
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
from numpy.linalg import inv


# Define Q function and Value function
def Q(beta, A1, A2, covariates):
    return np.dot(np.concatenate((A1, A2, A1*A2, covariates), axis=1), beta)

# Value function(V)
def V(tar_pi_A1, tar_pi_A2, covariates, beta):  # Treat original V_pi(St)
    V_St = Q(beta, tar_pi_A1, tar_pi_A2, covariates)
    return V_St

def OLS(data, X_init, tar_Pi_arr):
    # calculate the causal effect
    X = data[:, 0:7]
    A1 = data[:, 8].reshape(-1, 1)
    A2 = data[:, 9].reshape(-1, 1)
    Y0 = data[:, 10].reshape(-1, 1)
    Y12 = data[:, 11].reshape(-1, 1)
    Y24 = data[:, 12].reshape(-1, 1)
    Y36 = data[:, 13].reshape(-1, 1)

    # obtain the distribution of baseline covariates (X)
    X_test = X_init[:, 0:-1]

    # Estimate trajectories under the target policies
    tar_Pi_A1 = tar_Pi_arr['A1']
    tar_Pi_A2 = tar_Pi_arr['A2']

    # get estimate of Y0 and Y12 using linear regression
    Linear_model_Y0 = LinearRegression().fit(X, Y0)
    Linear_model_Y12 = LinearRegression().fit(np.hstack((X, Y0, A1)), Y12)
    Linear_model_Y24 = LinearRegression(fit_intercept=False).fit(np.hstack((A1, A2, A1 * A2, X, Y0, Y12)), Y24)
    Linear_model_Y36 = LinearRegression(fit_intercept=False).fit(np.hstack((A1, A2, A1 * A2, X, Y0, Y12)), Y36)

    Y0_esti = Linear_model_Y0.predict(X_test)
    Y12_esti = Linear_model_Y12.predict(np.hstack((X_test, Y0_esti, tar_Pi_A1)))
    beta_Q_24 = Linear_model_Y24.coef_.T
    beta_Q_36 = Linear_model_Y36.coef_.T


    covariates = np.concatenate((X_test, Y0_esti, Y12_esti), axis=1)

    # Estimating Y_24week using the Q function
    Y24_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_24)
    # Estimating Y_36week using the Q function
    Y36_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_36)

    out = np.concatenate([X_init, tar_Pi_A1, tar_Pi_A2, Y0_esti, Y12_esti, Y24_esti, Y36_esti], axis=-1)
    return out

def Proxy(data, X_init, tar_Pi_arr):
    # calculate the causal effect
    X = data[:, 0:7]
    A1 = data[:, 8].reshape(-1, 1)
    A2 = data[:, 9].reshape(-1, 1)
    Y0 = data[:, 10].reshape(-1, 1)
    Y12 = data[:, 11].reshape(-1, 1)
    Y24 = data[:, 12].reshape(-1, 1)
    Y36 = data[:, 13].reshape(-1, 1)
    Z1 = data[:, 14].reshape(-1, 1)
    Z2 = data[:, 15].reshape(-1, 1)
    W = data[:, 16].reshape(-1, 1)

    Linear_model_W = LinearRegression().fit(np.hstack((Z1, Z2, A1, A2)), W)
    hatW = Linear_model_W.predict(np.hstack((Z1, Z2, A1, A2)))

    Linear_model_Y0 = LinearRegression().fit(X, Y0)
    Linear_model_Y12 = LinearRegression().fit(np.hstack((X, Y0, A1)), Y12)
    Linear_model_Y24 = LinearRegression(fit_intercept=False).fit(np.hstack((A1, A2, A1 * A2, X, Y0, Y12, hatW)), Y24)
    Linear_model_Y36 = LinearRegression(fit_intercept=False).fit(np.hstack((A1, A2, A1 * A2, X, Y0, Y12, hatW)), Y36)

    # Estimate trajectories under the target policies
    tar_Pi_A1 = tar_Pi_arr['A1']
    tar_Pi_A2 = tar_Pi_arr['A2']

    # obtain the distribution of baseline covariates (X)
    X_test = X_init[:, 0:-1]

    Y0_esti = Linear_model_Y0.predict(X_test)
    Y12_esti = Linear_model_Y12.predict(np.hstack((X_test, Y0_esti, tar_Pi_A1)))

    beta_Q_24 = Linear_model_Y24.coef_[:,:-1].T
    beta_Q_36 = Linear_model_Y36.coef_[:,:-1].T

    covariates = np.concatenate((X_test, Y0_esti, Y12_esti), axis=1)

    # Estimating Y_24week using the Q function
    Y24_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_24)
    # Estimating Y_36week using the Q function
    Y36_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_36)

    out = np.concatenate([X_init, tar_Pi_A1, tar_Pi_A2, Y0_esti, Y12_esti, Y24_esti, Y36_esti], axis=-1)
    return out

def IV(data, X_init, tar_Pi_arr):
    # calculate the causal effect
    X = data[:, 0:7]
    A1 = data[:, 8].reshape(-1, 1)
    A2 = data[:, 9].reshape(-1, 1)
    Y0 = data[:, 10].reshape(-1, 1)
    Y12 = data[:, 11].reshape(-1, 1)
    Y24 = data[:, 12].reshape(-1, 1)
    Y36 = data[:, 13].reshape(-1, 1)
    Z1 = data[:, 14].reshape(-1, 1)
    Z2 = data[:, 15].reshape(-1, 1)

    Linear_model_A1 = LinearRegression().fit(Z1, A1)
    Linear_model_A2 = LinearRegression().fit(Z2, A2)

    hat_A1 = Linear_model_A1.predict(Z1)
    hat_A2 = Linear_model_A2.predict(Z2)

    Linear_model_Y0 = LinearRegression().fit(X, Y0)
    Linear_model_Y12 = LinearRegression().fit(np.hstack((X, Y0, A1)), Y12)
    Linear_model_Y24 = LinearRegression(fit_intercept=False).fit(np.hstack((hat_A1, hat_A2, A1 * A2, X, Y0, Y12)), Y24)
    Linear_model_Y36 = LinearRegression(fit_intercept=False).fit(np.hstack((hat_A1, hat_A2, A1 * A2, X, Y0, Y12)), Y36)

    # Estimate trajectories under the target policies
    tar_Pi_A1 = tar_Pi_arr['A1']
    tar_Pi_A2 = tar_Pi_arr['A2']

    # obtain the distribution of baseline covariates (X)
    X_test = X_init[:, 0:-1]

    Y0_esti = Linear_model_Y0.predict(X_test)
    Y12_esti = Linear_model_Y12.predict(np.hstack((X_test, Y0_esti, tar_Pi_A1)))

    beta_Q_24 = Linear_model_Y24.coef_.T
    beta_Q_36 = Linear_model_Y36.coef_.T

    covariates = np.concatenate((X_test, Y0_esti, Y12_esti), axis=1)

    # Estimating Y_24week using the Q function
    Y24_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_24)
    # Estimating Y_36week using the Q function
    Y36_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_36)

    out = np.concatenate([X_init, tar_Pi_A1, tar_Pi_A2, Y0_esti, Y12_esti, Y24_esti, Y36_esti], axis=-1)
    return out

def Aux(data, X_init, tar_Pi_arr):

    # calculate the causal effect
    X = data[:, 0:7]
    A1 = data[:, 8].reshape(-1, 1)
    A2 = data[:, 9].reshape(-1, 1)
    A1A2 = A1 * A2 # Consider A1A2 as the third action
    A = data[:,8:10]
    Y0 = data[:, 10].reshape(-1, 1)
    Y12 = data[:, 11].reshape(-1, 1)
    Y24 = data[:, 12].reshape(-1, 1)
    Y36 = data[:, 13].reshape(-1, 1)
    # only use one of IVs
    # Z1 = data[:, 14].reshape(-1, 1)
    Z1 = data[:, 15].reshape(-1, 1)


    # (1) regress action(A) on IV(Z)
    Linear_model_A = LinearRegression().fit(Z1, A)
    heta = Linear_model_A.coef_

    # regress outcome(Y_) on IV(Z), action(A) and covarites (X,Y0,Y12)
    Linear_model_Y0 = LinearRegression().fit(X, Y0)
    Linear_model_Y12 = LinearRegression().fit(np.hstack((X, Y0, A1)), Y12)
    Linear_model_Y24 = LinearRegression(fit_intercept=False).fit(np.hstack((A, A1A2, Z1, X, Y0, Y12)), Y24) # intercepts=false because there are intercepts in X.
    Linear_model_Y36 = LinearRegression(fit_intercept=False).fit(np.hstack((A, A1A2, Z1, X, Y0, Y12)), Y36)
    hxi_Y24 = Linear_model_Y24.coef_
    hxi_Y36 = Linear_model_Y36.coef_

    # (2) Factor analysis
    A_res = Linear_model_A.predict(Z1) - A
    fa = FactorAnalyzer(rotation=None, n_factors=1)
    fa.fit(A_res)
    efa_fit = fa.loadings_

    # (3) Calculate halpha_A
    var_A_res = np.var(A_res, axis=0, ddof=1)
    halpha_A = efa_fit * np.sqrt(var_A_res).reshape(-1, 1)

    # (4) Calculate hgamma
    cov_matrix = np.cov(A_res, rowvar=False)
    hgamma = inv(cov_matrix) @ halpha_A

    # (5) Estimate confounding bias (coefs of unobserved confounders)
    halpha_Y24 = -inv(hgamma.T @ heta @ heta.T @ hgamma) @ (hgamma.T @ heta) @ hxi_Y24[:, 3: 4]
    halpha_Y36 = -inv(hgamma.T @ heta @ heta.T @ hgamma) @ (hgamma.T @ heta) @ hxi_Y36[:, 3: 4]

    # (6) Estimate effects (the coefs of actions(A) on Y24 and Y36)
    hbeta_Y24 = hxi_Y24[:, :2] - (hgamma @ halpha_Y24).T
    hbeta_Y36 = hxi_Y36[:, :2] - (hgamma @ halpha_Y36).T


    # Estimate trajectories under the target policies
    tar_Pi_A1 = tar_Pi_arr['A1']
    tar_Pi_A2 = tar_Pi_arr['A2']

    # obtain the distribution of baseline covariates (X)
    X_test = X_init[:, 0:-1]

    Y0_esti = Linear_model_Y0.predict(X_test)
    Y12_esti = Linear_model_Y12.predict(np.hstack((X_test, Y0_esti, tar_Pi_A1)))

    # Y24_esti = (tar_Pi_A1 * hbeta_Y24[:,0] + tar_Pi_A2 * hbeta_Y24[:,1] + tar_Pi_A1 * tar_Pi_A2 * hxi_Y24[:,2] +
    #             np.dot(X_test, hxi_Y24[:, 4:11].T) + Y0_esti * hxi_Y24[:, 11] + Y12_esti * hxi_Y24[:, 12])
    # Y36_esti = (tar_Pi_A1 * hbeta_Y36[:,0] + tar_Pi_A2 * hbeta_Y36[:,1] + tar_Pi_A1 * tar_Pi_A2 * hxi_Y36[:,2] +
    #             np.dot(X_test, hxi_Y36[:, 4:11].T) + Y0_esti * hxi_Y36[:, 11] + Y12_esti * hxi_Y36[:, 12])

    beta_Q_24 = np.hstack([hbeta_Y24, hxi_Y24[:,2:3], hxi_Y24[:, 4:13]]).reshape(-1, 1)
    beta_Q_36 = np.hstack([hbeta_Y36, hxi_Y36[:,2:3], hxi_Y36[:, 4:13]]).reshape(-1, 1)

    covariates = np.concatenate((X_test, Y0_esti, Y12_esti), axis=1)

    # Estimating Y_24week using the Q function
    Y24_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_24)
    # Estimating Y_36week using the Q function
    Y36_esti = V(tar_Pi_A1, tar_Pi_A2, covariates, beta_Q_36)
    out = np.concatenate([X_init, tar_Pi_A1, tar_Pi_A2, Y0_esti, Y12_esti, Y24_esti, Y36_esti], axis=-1)
    return out

if __name__ == '__main__':
    np.random.seed(1)

    reps = 100 # number of replication
    sample_sizes = [100, 1000]  # Two sample sizes
    theta_values = [0.2, 0.5, 0.8]  # Theta values
    gamma_values = [1, 5, 10]  # Gamma values

    for sample_size in sample_sizes:
        for theta in theta_values:
            print(f'Sample size: {sample_size}, Theta: {theta}, Gamma: {1}')
            config_data = {'sigma': 1, 'theta': theta, 'Gamma': 1}
            Generator = DataGen(config=config_data)

            # get groundtruth
            Truth = Generator.get_groundtruth()
            # get target policy
            tar_Pi_arr = Generator.tar_Pi(sample_size)
            # get data
            data, X_init = Generator.Gen(sample_size)

            esti_OLS = np.zeros((sample_size, 14))
            esti_IV = np.zeros((sample_size, 14))
            esti_Aux = np.zeros((sample_size, 14))
            esti_Pi = np.zeros((sample_size, 14))

            for i in range(reps):
                # print(f'Iteration: {i + 1}')

                esti_OLS += OLS(data, X_init, tar_Pi_arr)
                esti_IV += IV(data, X_init, tar_Pi_arr)
                esti_Aux += Aux(data, X_init, tar_Pi_arr)
                esti_Pi += Proxy(data, X_init, tar_Pi_arr)

            esti_OLS /= reps
            esti_IV /= reps
            esti_Aux /= reps
            esti_Pi /= reps

            esti_list = [Truth, esti_OLS, esti_IV, esti_Aux, esti_Pi]
            plot_autism(esti_list, title='Autism simulation')

        for gamma in gamma_values:
            print(f'Sample size: {sample_size}, Theta: {0.95}, Gamma: {gamma}')
            config_data = {'sigma': 1, 'theta': 0.95, 'Gamma': gamma}
            Generator = DataGen(config=config_data)

            # get groundtruth
            Truth = Generator.get_groundtruth()
            # get target policy
            tar_Pi_arr = Generator.tar_Pi(sample_size)
            # get data
            data, X_init = Generator.Gen(sample_size)

            esti_OLS = np.zeros((sample_size, 14))
            esti_IV = np.zeros((sample_size, 14))
            esti_Aux = np.zeros((sample_size, 14))
            esti_Pi = np.zeros((sample_size, 14))

            for i in range(reps):
                # print(f'Iteration: {i + 1}')

                esti_OLS += OLS(data, X_init, tar_Pi_arr)
                esti_IV += IV(data, X_init, tar_Pi_arr)
                esti_Aux += Aux(data, X_init, tar_Pi_arr)
                esti_Pi += Proxy(data, X_init, tar_Pi_arr)

            esti_OLS /= reps
            esti_IV /= reps
            esti_Aux /= reps
            esti_Pi /= reps

            esti_list = [Truth, esti_OLS, esti_IV, esti_Aux, esti_Pi]
            plot_autism(esti_list, title='Autism simulation')




