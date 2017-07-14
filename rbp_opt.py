from __future__ import division
from qrel_chk import Qrel
import threading
import futils
import sys
import getopt
from collections import defaultdict
import math
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize

class RBPOpt (threading.Thread):
    """
    optimizing RBP to d value.
    The RMSE is calculated per run, for the final score.
    only on document that are judged.
    minimizing the ``fitted'' guessing value
    and the real judged ones.
    """

    def __init__(self, p, d, q, qrelname,
                 fitted_vec, rank_dir,
                 method, out_dir, is_binary=True):
        """
        init the opt process
        :param p: persistance values
        :param d: considered pooling depth.
        :param q: qid.
        :param qrelname: qrel name
        :param fitted_vec: fitted_vector for method
        :param rank_dir: dir of rank mat
        :param method: method idx
        :param out_dir: output dir
        :param: is_binary: True
        """
        threading.Thread.__init__(self)
        self._outname = out_dir + "opt-weight-"+str(method)+".txt"
        self._rmse = out_dir + "opt-rmse-" + str(method) + ".txt"
        self._k = d
        self._q = q
        self._qrel = Qrel(qrelname).get_rel_by_qid(q)
        self._p = p
        tmp_rank_mat, self._runnum = futils.read_csv_to_dict(rank_dir + str(q) + "-rank.txt", is_prob=False)
        self._rank_bg = fitted_vec
        self._rbp = np.zeros(self._runnum)
        self._bg_vectors = np.zeros((self._k, self._runnum, self._runnum))
        self._bg_rbp = np.zeros((self._k, self._runnum))
        self._binary = is_binary
        # load the rank matrix
        for k, v in tmp_rank_mat.iteritems():
            tmp_v = np.array(v)  # convert to np array for processing.
            is_judged = False
            curr_rel = 0
            if k in self._qrel:
                if self._qrel[k] > 0:
                    curr_rel = 1 if self._binary else self._qrel[k]
            if min(tmp_v) < self._k and max(tmp_v) > -1:  # this document is retrieved by one of the system
                tmp = self._rank_bg[tmp_v]
                for i in range(0, len(tmp_v)):
                    if 0 <= tmp_v[i] < self._k:
                        self._rbp[i] += curr_rel * math.pow(self._p, tmp_v[i])
                        self._bg_rbp[tmp_v[i], i] = curr_rel * math.pow(self._p, tmp_v[i])
                        if is_judged:
                            self._bg_vectors[tmp_v[i], i, :] = tmp  # set the fitted vector to judged documents

    def _est_func(self, w):
        """
        object function for the optimization.
        for optimizing score and using linear combination
        :param method:
        :return:
        """
        fitted_rbp = np.zeros(len(w))
        k = self._bg_vectors.shape[0]
        for i in range(0, k):  # iterating down to the rank
            curr_rank = self._bg_vectors[i, :, :]
            for j in range(0, self._runnum):
                fitted_rbp[j] += math.pow(self._p, i) * np.dot(w, curr_rank[j, :]) # get to the real bg of the document
        return LA.norm(fitted_rbp-self._rbp)

    def get_weight(self):
        w0 = np.ones(self._runnum) / self._runnum
        cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
        # For linear and per score
        bnds = tuple((0,1) for x in w0)
        res = minimize(self._est_func, w0, method='SLSQP', constraints=cons, bounds=bnds)
        # --------------------------------------
        res_str = str(self._q)
        for i in range(0,len(res.x)):
            res_str += ",{:.6f}".format(res.x[i])
        with open(self._outname, 'a') as fout:
            fout.write(res_str.strip() + "\n")
        with open(self._rmse,'a') as fout:
            fout.write(str(self._q) + ", {:.4f}".format(res.fun).strip() + "\n")

    def run(self):
        self.get_weight()


def get_doc_prob(qid, out_dir, rank_dir, fitted_dir, m = 4):
    """
    output final estimation based on the weighting param
    :param qrelname: qrel name
    :param out_dir: output dir, same as the previous used one
    :param rank_dir: rank-mat dir
    :param fitted_dir: fitted vector dir
    :param m: number of method
    :return:
    """
    runnum = 100
    param_mat = np.zeros((m, len(qid), runnum)) # shrink later
    for i in range(0, m):
        curr_mat =  np.loadtxt(out_dir + "opt-weight-"+str(i+1)+".txt", delimiter=",", dtype=float)
        if runnum >= curr_mat.shape[1]:
            runnum = curr_mat.shape[1] - 1
            param_mat = param_mat[:, :, 0:runnum]
        param_mat[i, :, :] = curr_mat[:, 1:]
    for q in range(0, len(qid)):
        doc_prob = defaultdict(list)
        rank_mat, runnum = futils.read_csv_to_dict(rank_dir + str(qid[q]) + "-rank.txt", is_prob=False)
        fit_mat = np.loadtxt(fitted_dir + str(qid[q]) + ".txt", delimiter=" ", dtype=float)
        for doc , rank in rank_mat.iteritems():
            if doc not in doc_prob:
                doc_prob[doc] = [0] * m
            for i in range(0, m):
                curr_gain = fit_mat[:, i + 1]
                doc_prob[doc][i] = np.dot(param_mat[i, q, :], curr_gain[np.array(rank)])
        with open(out_dir + str(qid[q]) + "-prob.txt", "a") as fout:
            for k,v in doc_prob.items():
                curr_str = str(k)
                for p in v:
                    curr_str += ", {:.4f}".format(p)
                fout.write(curr_str.strip() + "\n")

def main(argv):
    method = 3
    qrelfile = ""
    depth = 10
    collection = "rob04"
    # pd = 100
    # qid = 651
    try:
        opts, args = getopt.getopt(argv, "j:d:c:h", ["jfile", "depth", "collection"])
    except getopt.GetoptError:
        print('-j <qrelfile> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-j <qrelfile> -c <collection> -d <depth> -h help')
            sys.exit()
        elif opt in ("-j", "--jfile"):
            qrelfile = arg
        elif opt in ("-d", "--d"):
            depth = int(arg)
        elif opt in ("-c", "--c"):
            collection = arg
    # if collection == "tb06":
    #     pd = 50
    # elif collection == "tb04":
    #     pd = 80

    prifix_dir = "testcase/"
    rank_dir = prifix_dir + collection + "/doc_rank/"
    fit_dir = prifix_dir + collection +"/background_gain/fit/origin/"+str(depth) + "/"
    out_dir = prifix_dir+ collection+ "/background_gain/sample_rbp/opt_score/"+str(depth) + "/"

    curr_qrel = Qrel(qrelfile)
    qid = curr_qrel.get_qid()
    tlist = []
    p = 0.95
    for q in qid:
        fit_mat = np.loadtxt(fit_dir+str(q)+".txt", delimiter=" ", dtype=float)
        for i in range(1, 5):
            curr_opt = RBPOpt(p, 1000, q, qrelfile, fit_mat[:, i], rank_dir, i, out_dir)
            curr_opt.start()
            tlist.append(curr_opt)
        for t in tlist:
            t.join()
    get_doc_prob(qid, out_dir, rank_dir, fit_dir)
if __name__ == "__main__":
    main(sys.argv[1:])
