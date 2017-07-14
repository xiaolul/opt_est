from __future__ import division
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import getopt
import threading
from lmfit import Model


def const_model(y):
    """
    constant model
    if k \leq cutoff, bg is used, else
    bakground gain is zero
    :param y: observed probability
    :return:[k,p]: [cutoff depth,  background probability]
    """
    curr_param = None
    curr_min = 1000
    # piecewise will suffer local optima,
    # so we can only iterating all, brute-force
    for k in range(0, len(y)):
        for i in range(0, 1000):
            p = 0.001 * i
            tmp_x = np.zeros(len(y))
            tmp_x[0:k] = p
            perr = np.linalg.norm(tmp_x - y) / np.sqrt(len(y))
            if perr < curr_min:
                curr_param = [k, p]
                curr_min = perr
    return curr_param


def linear_model( k, lambda0, c):
    """
    linear model
    :param k: rank list
    :param lambda0: see equation
    :param c: see equation
    :return:
    """
    return lambda0 * k + c


def zipf_model( k, c, lambda0):
    """
    The zipf model to be fitted.
    lambda0/(i^c \cdot H_{1000,c})
    :param k: see equation
    :param c: see equation
    :param lambda0: scaling factor
    :return:
    """
    hsum = np.sum(np.array([np.power(1 / (1 + x), c) for x in range(0, 1000)]))
    return lambda0 / (np.power(k, c) * hsum)


def weibull_model( k, c, a, b):
    """
    Discrete Weibull. This is another form, with little difference
    but computationally friendly
    e^{-(x/a)^b}-e^{-(x+1/a)^b}
    :param k: rank vector
    :param c: scaling factor
    :param a: (alpha) see equation
    :param b: (alpha) see equation
    :return:
    """
    prev_k = np.exp(-1 * np.power(k / a, b))
    curr_k = np.exp(-1 * np.power((k + 1) / a, b))
    return c * (prev_k - curr_k)

class RelModels(threading.Thread):
    """
    Threading class for models, only for speed concern.
    especially when leave one out and
    randomization is added into the process.
    """
    def __init__(self, method, rel_mat,
                 kmax= 1000, is_random=False):
        """
        init probability model fitting class
        :param method, str: the model to be fitted
        :param rel_mat: observed relevance matrix
        :param kmax: the maximum predicted depth.
        :param is_random: do we need to perform randomization?
        """
        threading.Thread.__init__(self)
        self._relmat = rel_mat
        self._pred_x = kmax
        self._is_random = is_random
        self._pred_y = np.zeros(kmax)
        self._curr_method = method
        # get system numnbers for randomization
        self._rnum = rel_mat.shape[1]
        # consider 60% systems as the sample size
        self._csys = int(0.6* self._rnum)
        # how many times to perform the sampling
        self._shuffle_time = 100

    def _const_fit(self):
        """
        fitting constatnt model and set
        to the pred_y
        :return: None
        """
        if not self._is_random:
            curr_y = np.mean(self._relmat, axis= 1)
            k, p = const_model(curr_y)
            self._pred_y = np.zeros(self._pred_x)
            self._pred_y[0:k] = p
        else:
            for i in range(0, self._shuffle_time):
                curr_idx = np.random.choice(self._rnum, self._csys, replace=False)
                curr_relmat = self._relmat[:, curr_idx]
                curr_y = np.mean(curr_relmat, axis= 1)
                k, p = const_model(curr_y)
                tmp_y = np.zeros(self._pred_x)
                tmp_y[0:k] = p
                self._pred_y += tmp_y
            self._pred_y /=  self._shuffle_time

    def _linear_fit(self):
        """
        fit the linear model
        :return:
        """
        if not self._is_random:
            curr_avg = np.mean(self._relmat, axis=1)
            x_val = np.array(range(0, len(curr_avg))) + 1
            curr_model = Model(linear_model)
            curr_model.set_param_hint('lambda0', value=-0.1, min= -np.inf, max=0)
            curr_model.set_param_hint('c', value=1, min=0, max=np.inf)
            params = curr_model.make_params()
            result = curr_model.fit(curr_avg, k=x_val,
                                    params=params, method="leastsq")
            pred_x = np.array(range(0, self._pred_x)) + 1
            self._pred_y = result.eval(params = result.params, k = pred_x)
        else:
            for i in range(0, self._shuffle_time):
                curr_idx = np.random.choice(self._rnum, self._csys, replace=False)
                curr_relmat = self._relmat[:, curr_idx]
                curr_avg = np.mean(curr_relmat, axis=1)
                x_val = np.array(range(0, len(curr_avg))) + 1
                curr_model = Model(linear_model)
                curr_model.set_param_hint('lambda0', value=-0.1, min=-np.inf, max=0)
                curr_model.set_param_hint('c', value=1, min=0, max=np.inf)
                params = curr_model.make_params()
                result = curr_model.fit(curr_avg, k=x_val,
                                        params=params, method="leastsq")
                pred_x = np.array(range(0, self._pred_x)) + 1
                self._pred_y += result.eval(params=result.params, k=pred_x)
            self._pred_y /= self._shuffle_time

    def _zipf_fit(self):
        """
        fit the Zipf model
        :return:
        """
        if not self._is_random:
            curr_avg = np.mean(self._relmat, axis=1)
            x_val = np.array(range(0, len(curr_avg))) + 1
            curr_model = Model(zipf_model)
            curr_model.set_param_hint('c', value=1, min=0, max=np.inf)
            curr_model.set_param_hint('lambda0', value=1, min=0, max=np.inf)
            params = curr_model.make_params()
            result = curr_model.fit(curr_avg, k=x_val,
                                    params=params, method="leastsq")
            pred_x = np.array(range(0, self._pred_x)) + 1
            self._pred_y = result.eval(params=result.params, k=pred_x)
        else:
            for i in range(0, self._shuffle_time):
                curr_idx = np.random.choice(self._rnum, self._csys, replace=False)
                curr_relmat = self._relmat[:, curr_idx]
                curr_avg = np.mean(curr_relmat, axis=1)
                x_val = np.array(range(0, len(curr_avg))) + 1
                curr_model = Model(zipf_model)
                curr_model.set_param_hint('c', value=1, min=0, max=np.inf)
                curr_model.set_param_hint('lambda0', value=1, min=0, max=np.inf)
                params = curr_model.make_params()
                result = curr_model.fit(curr_avg, k=x_val,
                                        params=params, method="leastsq")
                pred_x = np.array(range(0, self._pred_x)) + 1
                self._pred_y += result.eval(params=result.params, k=pred_x)
            self._pred_y /= self._shuffle_time

    def _weibull_fit(self):
        """
        fit the Weibull Model
        Be careful that the Weibull starts with zero !
        :return:
        """
        if not self._is_random:
            curr_avg = np.mean(self._relmat, axis=1)
            x_val = np.array(range(0, len(curr_avg)))
            curr_model = Model(weibull_model)
            curr_model.set_param_hint('c', value = 1, min = 0, max = np.inf)
            curr_model.set_param_hint('a', value=1, min=0, max=np.inf)
            curr_model.set_param_hint('b', value=1, min=0, max=np.inf)
            pars = curr_model.make_params()
            result = curr_model.fit(curr_avg,
                              k = x_val,
                              params=pars,method="leastsq")
            pred_x = np.array(range(0, self._pred_x))
            self._pred_y = result.eval(params=result.params, k=pred_x)
        else:
            for i in range(0, self._shuffle_time):
                curr_idx = np.random.choice(self._rnum, self._csys, replace=False)
                curr_relmat = self._relmat[:, curr_idx]
                curr_avg = np.mean(curr_relmat, axis=1)
                x_val = np.array(range(0, len(curr_avg))) + 1
                curr_model = Model(weibull_model)
                curr_model.set_param_hint('c', value=1, min=0, max=np.inf)
                curr_model.set_param_hint('a', value=1, min=0, max=np.inf)
                curr_model.set_param_hint('b', value=1, min=0, max=np.inf)
                pars = curr_model.make_params()
                result = curr_model.fit(curr_avg,
                                        k=x_val,
                                        params=pars, method="leastsq")
                pred_x = np.array(range(0, self._pred_x))
                self._pred_y += result.eval(params=result.params, k=pred_x)
            self._pred_y /= self._shuffle_time

    def run(self):
        """
        Thread thing
        {"STATIC": 0, "CONST": 1, "LINEAR": 2, "ZIPF": 3, "WEIBULL": 4}
        :return:
        """
        if self._curr_method == 0:
            self._pred_y = np.ones(self._pred_x) * 0.5
        elif self._curr_method == 1:
            self._const_fit()
        elif self._curr_method == 2:
            self._linear_fit()
        elif self._curr_method == 3:
            self._zipf_fit()
        elif self._curr_method == 4:
            self._weibull_fit()

    def pred_y(self):
        """
        get the predicated values
        :return:
        """
        return np.clip(self._pred_y, 0, 1)  #  Make sure the relevance range.

    def method(self):
        """
        get indx of current fitting mehtod
        :return:
        """
        return self._curr_method

class FitProb:
    """
    fit background probability based on
    different assumptions and the pooling depth
    """
    def __init__(self, dirstr, pd, fmethod, leaveidx = None):
        """
        class init
        :param dirstr: rank rel dir
        :param pd: considered pooling depth
        :param fmethod: dictionary of fitting method
        :param leaveidx: perform leave-one-out. index of cols
        """
        self._flist = [f for f in listdir(dirstr) if isfile(join(dirstr, f))]
        self._reldir = dirstr
        self._pd = pd
        self._out_idx = leaveidx
        self._max_k = 1000  # maximum considered rank
        self._fmthod = fmethod
        if leaveidx is None:
            self._is_out = False
        else:
            self._is_out = True

    def dump_observed_prob(self, out_pref):
        """
        output the observed backgroud gain
        :param out_pref: prefix of output dir
        :return:
        """
        for f in self._flist:
            rel_mat = np.loadtxt(fname=self._reldir+f, delimiter=",",dtype=int)
            bg_vec = np.mean(rel_mat,axis=1)
            out_name = out_pref + f.split("-")[0] + "-prob.txt"
            np.savetxt(fname=out_name, X=bg_vec,fmt="%.6f")

    def fit_model(self, curr_depth, out_pref, is_random = False):
        """
        fitt different mdoels
        :param curr_depth: current considered fitting depth
        :param out_pref: prefix of output dir
        :param is_random: Perform randomization
        :return:
        """
        for f in self._flist:
            rel_mat = np.loadtxt(fname=self._reldir+f, delimiter=",",dtype=int)
            fitted_mat = np.zeros((self._max_k, len(self._fmthod)))
            t_pool = []
            curr_mat = rel_mat[0:curr_depth, :]
            contrib_idx = range(0, rel_mat.shape[1])
            if self._is_out:
                contrib_idx = [v for i , v in enumerate(contrib_idx) if i not in self._out_idx]
                curr_mat = curr_mat[:, contrib_idx]
            for k, v in self._fmthod.iteritems():
                curr_fit = RelModels(v,curr_mat,
                                     kmax=self._max_k,
                                     is_random=is_random)
                curr_fit.start()
                t_pool.append(curr_fit)
                #!TODO Need to change back to multi-thread later
                curr_fit.join()
            # for t in t_pool:
            #     t.join()
            for i in range(0, len(self._fmthod)):
                fitted_mat[:, t_pool[i].method()] = t_pool[i].pred_y()
            # save fitted dat
            out_name = out_pref + str(curr_depth) + "/" + f.split("-")[0]
            out_name += "-r.txt" if is_random else ".txt"
            np.savetxt(fname=out_name, X=fitted_mat, fmt="%.4f")


def main(argv):
    collection = "robust"
    pd = 100
    is_random = False
    gname=""
    try:
        opts, args = getopt.getopt(argv, "c:d:g:sh",
                                   ["collection", "pdepth",
                                    "shuffle","help"])
    except getopt.GetoptError:
        print('-c <collection> -d <pdepth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-c <collection> -d <pdepth> -h help')
            sys.exit()
        elif opt in ("-c", "--collection"):
            collection = arg
        elif opt in ("-d", "--pdepth"):
            pd = int(arg)
        elif opt in ("-s", "--shuffle"):
            is_random = True
        elif opt in ("-g", "--gname"):
            gname = arg

    # dirstr = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
    #                    + collection + "/rank_rel/"
    # out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
    #                + collection + "/background_gain/fit/"
    # gname_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"\
    #             +collection+"/qrels/log_qrels/"

    #!todo provide function for generating matrix
    dirstr = "default dir for rank times relevance matrix"
    out_dir = "default ouput dir"
    gname_dir = "default LOG qrels"
    leave_out = []

    if gname is not "":
        out_dir += "leave-out/" + gname + "/"
        with open(gname_dir+"input."+gname+".txt", 'rb') as fin:
            for line in fin:
                leave_out.append(int(line.strip()))
    else:
        out_dir += "origin/"

    fit_method = {"STATIC": 0, "CONST": 1, "LINEAR": 2, "ZIPF": 3, "WEIBULL": 4}
    fit_prob = FitProb(dirstr=dirstr, pd=pd, fmethod=fit_method,leaveidx=leave_out)
    fit_prob.fit_model(pd, out_dir, is_random=is_random)
    # fit_prob.dump_observed_prob(out_dir+"observe/")

if __name__ == "__main__":
    main(sys.argv[1:])
