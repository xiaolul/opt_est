from __future__ import division
from qrel_chk import Qrel
import threading
import futils
import sys
import getopt
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize


class LogRBPOpt (threading.Thread):
    """
    optimizing RBP to d value.
    calculate the RMSE based on per document gain,
    not for a system evaluate to the end
    """

    def __init__(self, p, d, q, qrelname,
                 fitted_vec, rank_dir,
                 method, out_dir, gidx,
                 is_binary=True):
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
        self._gidx = np.array(gidx)
        self._outname = out_dir + "opt-weight-" + str(method) + ".txt"
        self._rmse = out_dir + "opt-rmse-" + str(method) + ".txt"
        self._k = d
        self._q = q
        self._qrel = Qrel(qrelname).get_rel_by_qid(q)
        self._p = p
        tmp_rank_mat, self._runnum = futils.read_csv_to_dict(rank_dir + str(q) + "-rank.txt", is_prob=False)
        self._runnum -= len(self._gidx)
        self._rank_bg = fitted_vec
        self._rbp = np.zeros(self._runnum)
        self._bg_vectors = np.zeros((self._k, self._runnum, self._runnum))
        self._bg_rbp = np.zeros((self._k, self._runnum))
        self._binary = is_binary
        # load the rank matrix
        for k, v in tmp_rank_mat.iteritems():
            tmp_v = np.array(v)  # convert to np array for processing.
            tmp_v = np.delete(tmp_v, self._gidx)
            is_judged = False
            curr_rel = 0
            if k in self._qrel:
                if self._qrel[k] > 0:
                    curr_rel = 1 if self._binary else self._qrel[k]
                is_judged = True
            if min(tmp_v) < self._k and max(tmp_v) > -1:  # this document is retrieved by one of the system
                tmp = self._rank_bg[tmp_v]
                for i in range(0, len(tmp_v)):
                    if 0 <= tmp_v[i] < self._k:
                        self._rbp[i] += curr_rel * np.power(self._p, tmp_v[i])
                        self._bg_rbp[tmp_v[i], i] = curr_rel * np.power(self._p, tmp_v[i])
                        if is_judged:
                            self._bg_vectors[tmp_v[i], i, :] = tmp  # set the fitted vector to judged documents

    def _est_func_perdoc(self, w):
        """
        object function for the optimization.
        :param method:
        :return:
        """
        min_val = 0
        for i in range(0, self._runnum):
            curr_rank = self._bg_vectors[:, i, :]
            curr_rbp = self._bg_rbp[:, i]
            tmp_val = 0
            for j in range(0, self._k):
                tmp_val += (np.power(self._p, j) * np.dot(w, curr_rank[j, :]) - curr_rbp[j]) ** 2
            min_val += np.sqrt(tmp_val)
        return min_val

    def get_weight_perdoc(self):
        w0 = np.ones(self._runnum)/self._runnum
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        bnds = tuple((0, 1) for x in w0)
        #
        res = minimize(self._est_func_perdoc, w0, method='SLSQP', constraints=cons, bounds=bnds)
        res_str = str(self._q)
        for i in range(0, len(res.x)):
            res_str += "," + str(round(res.x[i], 6))
        with open(self._outname, 'a') as fout:
            fout.write(res_str.strip() + "\n")
        with open(self._rmse, 'a') as fout:
            fout.write(str(self._q) + ", {:.4f}".format(res.fun).strip() + "\n")

    def run(self):
        self.get_weight_perdoc()


def get_doc_prob(qid, out_dir, rank_dir, fitted_dir, gidx, m = 4):
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
            tmp_rank = np.delete(np.array(rank), np.array(gidx))
            if max(tmp_rank) != -1:
                if doc not in doc_prob:
                    doc_prob[doc] = [0] * m
                for i in range(0, m):
                    curr_gain = fit_mat[:, i + 1]
                    doc_prob[doc][i] = np.dot(param_mat[i, q, :], curr_gain[tmp_rank])
        with open(out_dir + str(qid[q]) + "-prob.txt","a") as fout:
            for k,v in doc_prob.items():
                curr_str = str(k)
                for p in v:
                    curr_str += ", {:.4f}".format(p)
                fout.write(curr_str.strip() + "\n")

def main(argv):
    qrelfile = ""
    depth = 10
    collection = "robust"
    gname = ""

    try:
        opts, args = getopt.getopt(argv, "d:hc:g:", ["runf", "depth"])
    except getopt.GetoptError:
        print('-j <qrelfile> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-j <qrelfile> -c <collection> -d <depth> -h help')
            sys.exit()
        # elif opt in ("-j", "--jfile"):
        #     qrelfile = arg
        elif opt in ("-d", "--d"):
            depth = int(arg)
        elif opt in ("-c", "--c"):
            collection = arg
        elif opt in ("-g", "--g"):
            gname = arg

    if gname=="":
        exit(1)

    rank_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
                collection + "/doc_rank/"
    fit_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
                collection +"/background_gain/fit/leave-out/"+gname+"/"+str(depth) + "/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
                collection+"/background_gain/leave-out/opt_doc/"+\
              gname+"/"+str(depth) + "/"
    qrel_dir  = "/research/remote/petabyte/users/xiaolul/pool_probability/"+collection+"/qrels/log_qrels/"
    gidx = []
    with open(qrel_dir+"input."+gname+".txt", "rb") as fin:
        for line in fin:
            gidx.append(int(line.strip()))
    qrelfile = qrel_dir + str(depth) + "/qrels.input."+gname+".txt"
    curr_qrel = Qrel(qrelfile)
    qid = curr_qrel.get_qid()
    tlist = []
    for q in qid:
        fit_mat = np.loadtxt(fit_dir+str(q)+".txt", delimiter=" ", dtype=float)
        for i in range(1, 5):
            curr_opt = LogRBPOpt(0.95, 1000, q, qrelfile,
                                 fit_mat[:, i], rank_dir, i,
                                 out_dir, gidx)
            curr_opt.start()
            tlist.append(curr_opt)
        for t in tlist:
            t.join()
    get_doc_prob(qid, out_dir, rank_dir, fit_dir, gidx)

if __name__ == "__main__":
    main(sys.argv[1:])