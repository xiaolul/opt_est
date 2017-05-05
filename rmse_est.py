from __future__ import division
import numpy as np
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.cross_validation import *
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from sklearn import grid_search
import os.path
import sys
import getopt

class RmseEst:
    """
    estimate rmse based on gamma, sample size
    We donot normalize or standardize feature
    values, since we choose to use ensemble methods
    "covratio", "gamma", "est_rel", "uniq_rel", "tot_rel", "rmse"
    """
    def __init__(self, fname, out_dir):
        """
        init class
        :param feature_dir:
        """
        tmp_mat = np.loadtxt(fname=fname,delimiter=",",dtype=float)
        tmp_mat = tmp_mat[~np.isinf(tmp_mat).any(axis=1)]  #Inf doesn't need to be considered
        colnum  = tmp_mat.shape[1]
        self._last_col = colnum - 1
        self._feature = tmp_mat[:, 1:self._last_col]
        self._rmse = tmp_mat[:, -1]
        self._train_dat, self._test_dat, self._train_y, self._test_y = \
            train_test_split(self._feature, self._rmse, test_size = 0.4, random_state = 0)
        self._out_dir = out_dir
        self._regressor = GradientBoostingRegressor()

    def training_phase(self):
        """
        training phase
        :return:
        """
        if os.path.isfile(self._out_dir+"gamma-rmse-param.txt"):
            best_param = pickle.load(open(self._out_dir+"gamma-rmse-param.txt", "rb"))
            self._regressor.set_params(**best_param)
            self._regressor.fit(self._train_dat, self._train_y)
        else:
            r = 0.01 * np.array(range(10, 100, 10))
            parameters = {"n_estimators": list(range(10, 300, 20)), "learning_rate": r.tolist()}
            mdl_train = grid_search.GridSearchCV(cv=5, param_grid=parameters, estimator=self._regressor, n_jobs=5)
            mdl_train.fit(self._train_dat,self._train_y)
            pickle.dump(mdl_train.best_params_, open(self._out_dir+"gamma-rmse-param.txt", "w"))
            self._regressor.set_params(**mdl_train.best_params_)
            self._regressor.fit(self._train_dat, self._train_y)
            pred_valid = self._regressor.predict(self._test_dat)
            error_score = r2_score(pred_valid, self._test_y)
            mse = mean_squared_error(pred_valid, self._test_y)
            print (error_score, mse)

    def pred_rmse(self, fname, depth):
        """
        read the new feature matrix and predict rmse
        :param fname: input feature mat name
        :param c_name: collection name
        :return: return predicated rmse with query id
        """
        fmat = np.loadtxt(fname, delimiter=",", dtype=float)
        fmat = fmat[~np.isinf(fmat).any(axis=1)]
        qid = fmat[:, 0]
        res = np.zeros((len(qid), 2))
        res[:, 0] = qid
        fmat = fmat[:, 1:self._last_col]
        res[:, 1] = self._regressor.predict(fmat)
        out_name = self._out_dir + depth + "-pred-rmse.txt"
        np.savetxt(fname=out_name, X=res, delimiter=",", fmt="%.4f")


def main(argv):
    collection = "trec9"
    depth = 10
    try:
        opts, args = getopt.getopt(argv, "c:d:h", ["collection", "depth"])
    except getopt.GetoptError:
        print('-c <collection> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-c <collection> -d <depth> -h help')
            sys.exit()
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-c", "--depth"):
            collection = arg
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/naive_estimator/"
    pred_fname = out_dir + collection + "/feature-" + str(depth)+".txt"
    estimator = RmseEst(fname=out_dir+"training.txt", out_dir=out_dir)
    estimator.training_phase()
    estimator.pred_rmse(pred_fname, collection+"/"+ str(depth))

if __name__ == "__main__":
        main(sys.argv[1:])



