from re import S, X
from numba import jit, prange
import ctypes
from numba.extending import get_cython_function_address
from scipy import integrate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf


@jit(nopython=True)

def simulate_model_single(t_obs, alpha, beta, Lambda, s, Theta):
    """
    Simulate a dataset from the model
    describing dynamics of the number of individuals over time.

    Arguments:
    t_obs   : int -- length of the observed time-series
    alpha   : float -- the alpha parameter of the CustomModel model
    beta    : float -- the beta parameter of the CustomModel model
    Lambda  : float -- the lambda parameter of the RegenModel model
    s       : float -- the s parameter of the RegenModel model
    Theta   : float -- the theta parameter of the RegenModel model
    """

    alpha = alpha if alpha >= 0 else 0
    beta = beta if beta >= 0 else 0
    Lambda = Lambda if Lambda > 0 else 0
    s = s if s >= 0 else 0
    Theta = Theta if Theta >= 0 else 0 

  

    x = np.zeros(t_obs)
    x0 = np.zeros(t_obs)
    intNomCap = 1 # Inital Norminal Capacity 

    for k in range(t_obs):
        delta = (alpha * beta * k) / ( 1e4 + (k * beta) ** 2)
        Inv_Lambda = (1.0 / Lambda)
        offsets = np.ones(t_obs) * np.floor( np.random.random() * Inv_Lambda)
        intNomCap -= delta 
        x[k] = intNomCap 
        #x0[k] = 0
        
        if round(k % Inv_Lambda) == 0 and k != 0: # offsets[i]:                 
                sample = np.random.gamma(Theta,s) # This is selected only to run the code.         
                x0[k] = sample
                x[k] += x0[k]
                intNomCap = x[k]
        # Add noise
        x[k] += np.random.normal(0,1e-3) 


        if x[k] > 1: 
            x[k] = 1 
        if x[k] < 0:
            x[k] = 0 
        if intNomCap > 1: 
            intNomCap = 1   
        if intNomCap < 0: 
            intNomCap = 0             
           
    return x


@jit(nopython=True, cache=True, parallel=True)
def simulate_model_batch(X, params, n_batch, t_obs):
    """
    Simulates a batch of model processes in parallel.
    """
    for i in prange(n_batch):
        X[i, :] = simulate_model_single(t_obs, params[i, 0], params[i, 1], params[i, 2],params[i, 3],params[i, 4])
    return X 


def simulate_model(batch_size=64, n_points=170, t_obs_min=100, t_obs_max=500, low_alpha=0.7, high_alpha=1.7,
                    low_beta=2.1, high_beta=3.6, low_Lambda=0.05, high_Lambda=0.1, low_s=0.008, high_s=0.02, low_Theta=1.001, high_Theta=2, to_tensor=True):

                    # low_Lambda=0.02, high_Lambda=0.1, low_s=1.001, high_s=2.0,low_Theta=0.001, high_Theta=0.0175
    """ 
    Simulates and returns a batch of 1D timeseries obtained under the  model.
    ----------

    Arguments:
    batch_size  : int -- number of model processes to simulate
    n_points    : int -- length of the observed time-series
    low_alpha   : float -- lower bound for the uniform prior on alpha
    high_alpha  : float -- upper bound for the uniform prior on alpha
    low_beta    : float -- lower bound for the uniform prior on beta

    high_beta   : float -- upper bound for the uniform prior on beta
    low_Lambda  : float -- lower bound for the uniform prior on Lambda
    high_Lambda : float -- upper bound for the uniform prior on Lambda
    low_s       : float -- lower bound for the uniform prior on s
    high_s      : float -- upper bound for the uniform prior on s
    low_Theta   : float -- lower bound for the uniform prior on Theta
    high_Theta  : float -- upper bound for the uniform prior on Theta
    ----------

    Returns:
    (X, theta)  : (np.array of shape (batch_size, t_obs, 1), np.array of shape (batch_size, 3)) or
              (tf.Tensor of shape (batch_size, t_obs, 1), tf.Tensor of shape (batch_size, 3)) --
              a batch or time series generated under a batch of Ricker parameters
    """


    # Sample t_obs, if None given
    if n_points is None:
        n_points = np.random.randint(low=t_obs_min, high=t_obs_max+1)

    # Prepare placeholders
    theta = np.random.uniform(low=[low_alpha, low_beta, low_Lambda,low_s,low_Theta],
                          high=[high_alpha, high_beta, high_Lambda,high_s,high_Theta], size=(batch_size, 5))
    X = np.zeros((batch_size, n_points))

    # Simulate a batch from the  model
    X = simulate_model_batch(X, theta, batch_size, n_points) # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% simulate_model_batch(X, params, n_batch, t_obs):
 
    if to_tensor:
        # return tf.convert_to_tensor(X[:, :, np.newaxis], dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
        return tf.convert_to_tensor(X[:, :], dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    #np.save('model_X', X[:, :, np.newaxis])
    #np.save('model_theta', theta)
    #return X[:, :, np.newaxis], theta
    return X, theta


def simulate_model_params(theta, n_points=500, to_tensor=True):
    """
    Simulates a batch of model datasets given parameters.
    """

    theta = np.atleast_2d(theta)
    X = np.zeros((theta.shape[0], n_points))

    # Simulate a batch from the  model
    X = simulate_model_batch(X, theta, theta.shape[0], n_points)
     
    X = X[:, :]#X = X[:, :, np.newaxis]

    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32)
    return X



def plot_model_multiple(T=500, figsize=(8, 3), filename='model'): # ????????????????? No return
    """Plots example datasets from the  model."""

    X, theta = simulate_model(10, n_points=T, to_tensor=False)
    t = np.arange(1, T+1)

    f, axarr = plt.subplots(2, 5, figsize=figsize)

    for i, ax in enumerate(axarr.flat):

        sns.lineplot(t, X[i, :, 0], ax=ax)
        if i == 0:
            ax.set_xlabel(r"Generation number $t$", fontsize=10)
            ax.set_ylabel("Number of individuals", fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    f.tight_layout()

    if filename is not None:
        f.savefig("figures/{}_plot_multiple.png".format(filename), dpi=300, bbox_inches='tight')


def load_test_model(to_tensor=True):
    """
    A utility for loading the test data.
    """

    X_test = np.load('model_X.npy')
    theta_test = np.load('model_theta.npy')

    if to_tensor:
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        theta_test = tf.convert_to_tensor(theta_test, dtype=tf.float32)

    return X_test, theta_test
