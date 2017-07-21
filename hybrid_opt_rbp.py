from __future__ import division
from qrel_chk import Qrel
import futils
import sys
import getopt
from collections import defaultdict
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize
import threading


class HybridOpt(threading.Thread):
    """
    optimizing RBP to d value.
    For a single topic, and a set of contributing systems
    to calculating a min RMSE
    """

    def __init__(self, p, d, q,
                 qrels, out_dir, rank_dir,
                 method, is_binary=True):
        """
        init the hybrid process.
        :param p: rbp parameter
        :param d: evalutation depth. 1000.
        :param q: query id
        :param qrels: qrels Qrel
        :param out_dir: output dir
        :param rank_dir: rank mat dir
        :param method: (opt_score or opt_doc , depth)
        :param is_binary:
        """
        threading.Thread.__init__(self)
        self._method = method
        self._k = d
        self._q = q
        self._qrel = qrels
        self._p = p
        self._binary = is_binary
        # Get the first round evaluation and the number of fitting method considered
        self._estimate, self._mnum = futils.read_csv_to_dict(HybridOpt.get_dir_str(out_dir, method) +
                                                             str(q) + "-prob.txt", is_prob=True)
        self._rank_mat, self._runnum = futils.read_csv_to_dict(rank_dir +
                                                               str(q) + "-rank.txt", is_prob=False)
        self.res = None  # optimization results, weighting parameters
        self._mnum -= 1  # remove constant method
        self._doc_rel = np.zeros((self._k, self._runnum, self._mnum))
        self._rbp = np.zeros(self._runnum)
        self._bg_rbp = np.zeros((self._k, self._runnum))
        # # load the rank matrix
        for k, v in self._rank_mat.iteritems():
            tmp_v = np.array(v)
            curr_rel = 0
            is_judged = False
            if k in self._qrel:
                if self._qrel[k] > 0:
                    curr_rel = 1 if self._binary else self._qrel[k]
                is_judged = True
            if min(tmp_v) < self._k and max(tmp_v) > -1:  # this document is retrieved by one of the system
                for i in range(0, len(tmp_v)):
                    if 0 <= tmp_v[i] < self._k:
                        self._rbp[i] += curr_rel * np.power(self._p, tmp_v[i])
                        self._bg_rbp[tmp_v[i], i] = curr_rel * np.power(self._p, tmp_v[i])
                        if is_judged:
                            self._doc_rel[tmp_v[i], i, :] = np.array(self._estimate[k][1:])

    def _est_func(self, w):
        """
        object function for the optimization.
        for optimizing score and using linear combination
        :param method:
        :return:
        """
        fitted_rbp = np.zeros(self._runnum)
        for i in range(0, self._k):  # iterating down to the rank
            curr_run = self._doc_rel[i, :, :]
            for j in range(0, self._runnum):
                fitted_rbp[j] += np.power(self._p, i) * np.dot(curr_run[j, :], w)  # get to the real bg of the document
        return LA.norm(fitted_rbp - self._rbp)

    def _est_doc_func(self, w):
        min_val = 0
        for i in range(0, self._runnum):
            curr_rank = self._doc_rel[:, i, :]
            curr_rbp = self._bg_rbp[:, i]
            tmp_val = 0
            for j in range(0, self._k):
                tmp_val += (np.power(self._p, j) * np.dot(w, curr_rank[j, :]) - curr_rbp[j]) ** 2
            min_val += np.sqrt(tmp_val)
        return min_val

    def get_weight(self):
        w0 = np.zeros(self._mnum)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        bnds = tuple((0, 1) for x in w0)
        if self._method[1] == 0 or self._method[1] == 2:
            self.res = minimize(self._est_func, w0, method='SLSQP', constraints=cons, bounds=bnds)
        elif self._method[1] == 1 or self._method[1] == 3:
            self.res = minimize(self._est_doc_func, w0, method='SLSQP', constraints=cons, bounds=bnds)

    @staticmethod
    def get_dir_str(out_dir, method):
        if method[1] == 0:
            return out_dir + "opt_score/" + str(method[0]) + "/".strip()
        elif method[1] == 1:
            return out_dir + "opt_doc/" + str(method[0]) + "/".strip()
        elif method[1] == 2:
            return out_dir + "optg_score/"+str(method[0])+"/".strip()
        elif method[1] == 3:
            return out_dir + "optg_doc/"+str(method[0])+"/".strip()
        else:
            return out_dir + "hybrid_opt/"+str(method[0])+"/".strip()

    def run(self):
        self.get_weight()


def get_doc_prob(out_dir, res, q, depth):
    with open(HybridOpt.get_dir_str(out_dir, (depth, -1)) + "rmse.txt", 'a') as fout:
        curr_str = str(q)
        for i in range(0, len(res)):
            curr_str += ",{:.4f}".format(res[i].fun)
        fout.write(curr_str.strip() + "\n")
    with open(HybridOpt.get_dir_str(out_dir, (depth, -1)) + "param.txt", 'a') as fout:
        curr_str = str(q)
        for i in range(0, len(res)):
            curr_w = res[i].x
            for j in range(0, len(res[i].x)):
                curr_str += ",{:.4f}".format(curr_w[j])
        fout.write(curr_str.strip() + "\n")
    doc_prob = defaultdict(list)
    for i in range(0, len(res)):
        curr_dict, mnum = futils.read_csv_to_dict(HybridOpt.get_dir_str(out_dir, (depth, i)) +
                                                  str(q) + "-prob.txt", is_prob=True)
        mnum -= 1
        for k, v in curr_dict.iteritems():
            if k not in doc_prob:
                doc_prob[k] = []
            tmp_prob = np.array(curr_dict[k][1:])
            doc_prob[k].append(np.dot(res[i].x, tmp_prob))
    with open(HybridOpt.get_dir_str(out_dir, (depth, -1)) + str(q) + "-prob.txt", "a") as fout:
        for k, v in doc_prob.iteritems():
            curr_str = str(k)
            for j in range(0, len(v)):
                curr_str += ", {:.4f}".format(v[j])
            fout.write(curr_str.strip() + "\n")


def main(argv):
    qrelfile = ""
    depth = 10
    collection = "robust"
    # pd = 100

    try:
        opts, args = getopt.getopt(argv, "j:d:hc:", ["runf", "jfile", "depth"])
    except getopt.GetoptError:
        print('-r <runlist> -j <qrelfile> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-r <runlist> -j <qrelfile> -o <output> -d <depth> -h help')
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
    fit_dir = prifix_dir + collection + "/background_gain/fit/origin/" + str(depth) + "/"
    out_dir = prifix_dir + collection + "/background_gain/sample_rbp/hybrid/" + str(depth) + "/"

    curr_qrel = Qrel(qrelfile)
    result_list = [None] * 4
    t_list = []
    qid = curr_qrel.get_qid()
    w_param = [None] * 4
    for q in range(0, len(qid)):
        for i in range(0, 4):
            result_list[i] = HybridOpt(0.95, 1000, qid[q],
                                       curr_qrel.get_rel_by_qid(qid[q]),
                                       out_dir, rank_dir,
                                       (depth, i))
            result_list[i].start()
            t_list.append(result_list[i])
        for t in t_list:
            t.join()
        for i in range(0, 4):
            w_param[i] = result_list[i].res
        get_doc_prob(out_dir, w_param, qid[q], depth)


if __name__ == "__main__":
    main(sys.argv[1:])
