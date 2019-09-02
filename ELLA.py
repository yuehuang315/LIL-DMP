""" Alpha version of a version of ELLA that plays nicely with sklearn
	@author: Paul Ruvolo
"""

from math import log
import numpy as np
from scipy.misc import logsumexp
from scipy.linalg import sqrtm, inv, norm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, explained_variance_score


class ELLA(object):
    """ The ELLA model """
    def __init__(self, d, k, base_learner, base_learner_kwargs = {}, mu = 1, lam = 1, k_init = False):
        """ Initializes a new model for the given base_learner.
            d: the number of parameters for the base_learner
            k: the number of latent model components
            base_learner: the base learner to use (currently can only be
                LinearRegression, Ridge, or LogisticRegression).
            base_learner_kwargs: keyword arguments to base learner (for instance to
                                 specify regularization strength)
            mu: hyperparameter for sparsity
            lam: L2 penalty on L
            mu: the L_1 penalty to use
            lam: the L_2 penalty to use
            NOTE: currently only binary logistic regression is supported
        """
        self.d = d
        self.k = k
        self.L = np.random.randn(d,k)
        self.A = np.zeros((d * k, d * k))
        self.b = np.zeros((d * k, 1))
        self.S = np.zeros((k, 0))
        self.T = 0
        self.mu = mu
        self.lam = lam
        self.k_init = k_init
        if base_learner in [LinearRegression, Ridge]:
            self.perf_metric = explained_variance_score
        elif base_learner in [LogisticRegression]:
            self.perf_metric = accuracy_score
        else:
            raise Exception("Unsupported Base Learner")

        self.base_learner = base_learner
        self.base_learner_kwargs = base_learner_kwargs

    def fit(self, X, y, task_id):
        """ Fit the model to a new batch of training data.  The task_id must
            start at 0 and increase by one each time this function is called.
            Currently you cannot add new data to old tasks.

            X: the training data
            y: the trianing labels
            task_id: the id of the task
        """
        self.T += 1
        # Here single_task_model is the f function which input is X,y
        # To produce the parameter theta_t to fit X,y
        # L loss function: LinearRegression/Ridge
        single_task_model = self.base_learner(fit_intercept = False, **self.base_learner_kwargs).fit(X, y)
        D_t = self.get_hessian(single_task_model, X, y)
        D_t_sqrt = sqrtm(D_t)
        # 权重向量 coef_：array，shape（n_features，）或（n_targets，n_features
        # initial theta from f function regression x,y
        theta_t = single_task_model.coef_

        # gT(L) function
        sparse_encode = Lasso(alpha = self.mu * 100 / (X.shape[0] * 2.0),
                              fit_intercept = False, max_iter = 100000).fit(D_t_sqrt.dot(self.L),
                                                         D_t_sqrt.dot(theta_t.T))
        # For Lasso input
        # D_t_sqrt.dot(self.L) is x
        # D_t_sqrt.dot(theta_t.T) is y
        # Lasso's coef_ is S(t)

        if self.k_init and task_id < self.k:
            sparse_coeffs = np.zeros((self.k,))
            sparse_coeffs[task_id] = 1.0
        else:
            sparse_coeffs = sparse_encode.coef_
        # sparse_coeffs is s(t)
        self.S = np.hstack((self.S, np.matrix(sparse_coeffs).T))
        self.A += np.kron(self.S[:,task_id].dot(self.S[:,task_id].T), D_t)
        self.b += np.kron(self.S[:,task_id].T, np.mat(theta_t).dot(D_t)).T
        # update L
        L_vectorized = inv(self.A / self.T + self.lam * 
        	np.eye(self.d * self.k, self.d * self.k)).dot(self.b) / self.T
        self.L = L_vectorized.reshape((self.k, self.d)).T
        self.revive_dead_components()

    def revive_dead_components(self):
        """ re-initailizes any components that have decayed to 0 """
        for i,val in enumerate(np.sum(self.L, axis = 0)):
            if abs(val) < 10 ** -8:
                self.L[:, i] = np.random.randn(self.d,)

    def predict(self, X, task_id):
        """ Output ELLA's predictions for the specified data on the specified
            task_id.  If using a continuous model (Ridge and LinearRegression)
            the result is the prediction.  If using a classification model
            (LogisticRgerssion) the output is currently a probability.
        """
        if self.base_learner == LinearRegression or self.base_learner == Ridge:
            return X.dot(self.L.dot(self.S[:, task_id]))
        elif self.base_learner == LogisticRegression:
            return 1. / (1.0 + np.exp(-X.dot(self.L.dot(self.S[:, task_id])))) > 0.5

    def predict_probs(self, X, task_id):
        """ Output ELLA's predictions for the specified data on the specified
            task_id.  If using a continuous model (Ridge and LinearRegression)
            the result is the prediction.  If using a classification model
            (LogisticRgerssion) the output is currently a probability.
        """
        if self.base_learner == LinearRegression or self.base_learner == Ridge:
            raise Exception("This base learner does not support predicting probabilities")
        elif self.base_learner == LogisticRegression:
            return np.exp(self.predict_logprobs(X, task_id))

    def predict_logprobs(self, X, task_id):
        """ Output ELLA's predictions for the specified data on the specified
            task_id.  If using a continuous model (Ridge and LinearRegression)
            the result is the prediction.  If using a classification model
            (LogisticRgerssion) the output is currently a probability.
        """
        if self.base_learner == LinearRegression or self.base_learner == Ridge:
            raise Exception("This base learner does not support predicting probabilities")
        elif self.base_learner == LogisticRegression:
            return -logsumexp(np.hstack((np.zeros((X.shape[0], 1)), -X.dot(self.L.dot(self.S[:, task_id])))), axis = 1)

    def score(self, X, y, task_id):
        """ Output the score for ELLA's model on the specified testing data.
            If using a continuous model (Ridge and LinearRegression)
            the score is explained variance.  If using a classification model
            (LogisticRegression) the score is accuracy.
        """
        return self.perf_metric(self.predict(X, task_id), y)

    def get_hessian(self, model, X, y):
        """ ELLA requires that each single task learner provide the Hessian
            of the loss function evaluated around the optimal single task
            parameters.  This funciton implements this for the base learners
            that are currently supported """
        theta_t = model.coef_
        # calculate D(t)
        if self.base_learner == LinearRegression:
            return X.T.dot(X)/(2.0 * X.shape[0])
        elif self.base_learner == Ridge:
            return X.T.dot(X)/(2.0 * X.shape[0]) + model.alpha * np.eye(self.d, self.d)
        elif self.base_learner == LogisticRegression:
            preds = 1. / (1.0 + np.exp(-X.dot(theta_t.T)))
            base = np.tile(preds * (1 - preds), (1, X.shape[1]))
            hessian = (np.multiply(X, base)).T.dot(X) / (2.0 * X.shape[0])
            return hessian + np.eye(self.d,self.d) / (2.0 * model.C)