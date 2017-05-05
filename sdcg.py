from __future__ import division
from qres_chk import Qres
from qrel_chk import Qrel
import numpy as np
import os
import futils
import sys
import getopt


class SDCG:
    """
    calculate sdcg score
    with residual
    """
    def __init__(self, qrelname, qresname, b=2,
                 k=1000, max_rel = 1):
        """
        init the calculation of SDCG
        :param qrelname: qrel name
        :param qresname: qres name
        :param b: log base
        :param k: evaluation depth, 1000 by default
        :param is_binary: conisder binary judgment by default for now
        """
        self._qrel = Qrel(qrelname)
        self._qres = Qres(qresname)
        self._b = b if b > 2 else 2
        self._base =  np.log2(self._b) if self._b != 2 else 1
        self._k  = k
        self._is_binary = True if max_rel == 1 else False
        self._max_rel = max_rel
        self._qid = self._qrel.get_qid()
        self._default = 0.0
        self._rname = os.path.basename(qresname)
        self._gname = os.path.basename(qrelname).split(".")[2]
        self._qrel_dir = os.path.dirname(qrelname) + "/"
        self._scaling_factor = self._scaling_calc()

    def _scaling_calc(self):
        """
        calculate the maximum cumulated gain
        based on the discounting base and the maximum rel
        :return: the scaling factor
        """
        scaling_factor = 0.0
        for i in range(1, self._k):
            scaling_factor += self._base/np.log2(i + 1)
        return scaling_factor

    def sdcg_eval(self, out_dir):
        """
        Calculate the SDCG value
        :param: output dir
        :return:
        """
        score_str = [str(self._k) + "," + "b="+str(self._b) + ","] * len(self._qid)
        for q in range(0, len(self._qid)):
            curr_score = 0
            residual = 0
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            last_pos = self._k if len(curr_res) > self._k else len(curr_res)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = 1 if self._is_binary else curr_rel[curr_doc]
                        curr_score += rel_val * self._base / np.log2(i + 2)
                else:
                    residual += self._max_rel * self._base / np.log2(i + 2)
            # for i in range(last_pos, 1000):
            #     residual += self._max_rel * self._base / np.log2(i + 2)
            score_str[q] = str(self._qid[q]) + "," + score_str[q] + \
                           "s={:.4f},r={:.4f}".format(curr_score / self._scaling_factor,
                                                      residual / self._scaling_factor).strip()
        with open(out_dir + self._rname + ".sdcg", "w") as fout:
            for s_str in score_str:
                fout.write(s_str + "\n")

    def sdcg_fit_eval(self, prob_dir, out_dir, is_radom=False):
        """
        calculate the sdcg score based on thefitted background gain
        including the RM method.
        lb, [fitting method], RM, upper bound
        :param prob_dir:
        :param out_dir:
        :param is_radom:
        :return:
        """
        score = np.zeros((len(self._qid), 8))
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            fname = prob_dir + str(self._qid[q])
            if is_radom:
                fname += "-r.txt"
            else:
                fname += ".txt"
            prob_mat = np.loadtxt(fname, delimiter=" ", dtype=float)
            m_num = prob_mat.shape[1]
            if score.shape[1] > m_num + 3:
                score = score[:, (m_num + 3)]
            last_pos = len(curr_res) if len(curr_res) < self._k else self._k
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    rel = curr_rel[curr_doc]
                    if rel > 0:
                        rel_val = 1 if self._is_binary else curr_rel[curr_doc]
                        score[q, :] += rel_val * self._base / np.log2(i + 2)
                else:  # the document is not judged
                    score[q, -1] += self._max_rel * self._base / np.log2(i + 2)
                    score[q, 1:(m_num + 1)] += prob_mat[i, :] * self._base / (np.log2(i + 2))
            # for i in range(last_pos, 1000):  Do not use this
            #     score[q, -1] += self._max_rel * self._base / np.log2(i + 2)
            #     score[q, 1:(m_num + 1)] += prob_mat[i, :] * self._base / (np.log2(i + 2))
            score[q, :] /= self._scaling_factor
            # calculate the RM method
            delta = score[q, -1] - score[q, 0]
            if delta == 1:
                score[q, -2] = score[q, 0] + delta * 0.01
            else:
                score[q, -2] = score[q, 0] + delta * score[q, 0] / (1 - delta)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.sdcg"
        else:
            fname += ".sdcg"
        np.savetxt(fname=fname, X=score, fmt="%.4f", delimiter=",")

    def sdcg_est_eval(self, prob_dir, out_dir, is_radom=False):
        """
        evaluate score based on the estimated gain.
        per document, if the document has not been estimated,
        use the fitted gain instead.
        Notice that the residual will not be calculated in this case,
        everything is based on the estimated only.
        :param prob_dir: per document background gain dir
        :param out_dir: output dir
        :param is_radom: use randomized result?
        :return:
        """
        score = np.zeros((len(self._qid), 4))
        init_m = score.shape[1]
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            prob_name = prob_dir + str(self._qid[q])
            if is_radom:
                prob_name += "-r.txt"
            else:
                prob_name += "-prob.txt"
            prob_mat, mnum = futils.read_csv_to_dict(prob_name, is_prob=True)
            if init_m > mnum:   # shrink to fit
                init_m = mnum
            last_pos = self._k if len(curr_res) > self._k else len(curr_res)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = 1 if self._is_binary else curr_rel[curr_doc]
                        score[q, 0:init_m] += rel_val * self._base / np.log2(i + 2)
                else:  # use estimated values instead
                    est_rel = np.array(prob_mat[curr_doc])
                    score[q, 0:init_m] += est_rel * self._base / np.log2(i + 2)
            score[q, 0:init_m] /= self._scaling_factor
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.sdcg"
        else:
            fname += ".sdcg"
        np.savetxt(fname=fname, X=score[:, 0:init_m], fmt="%.4f", delimiter=",")


def main(argv):
    runfile = ""
    qrelfile = ""
    depth = 10
    collection = "robust"
    pd = 100
    is_log = False
    gname = ""
    is_random = False
    is_plain = False  # only perform the plain rbp calculation, based on qrel file
    try:
        opts, args = getopt.getopt(argv, "r:j:d:c:g:sh", ["runf", "jfile", "depth"])
    except getopt.GetoptError:
        print('-r <runlist> -j <qrelfile> -d <depth> -c <collection> -g <gname> -s shuffle -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-r <runlist> -j <qrelfile> -d <depth> -c <collection> -g <gname> -s shuffle -h help')
            sys.exit()
        elif opt in ("-r", "--runf"):
            runfile = arg
        elif opt in ("-j", "--jfile"):
            qrelfile = arg
            is_plain = True
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-c", "--depth"):
            collection = arg
        elif opt in ("-g", "--grp name"):
            gname = arg
            is_log = True
        elif opt in ("-s", "--shuffle"):
            is_random = True
    if collection == "tb06":
        pd = 50
    elif collection == "tb04":
        pd = 80

    if collection == "tb06":
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                   "/cluewebap/sigirexp/rank_order/original/trec2006/"+runfile
    else:
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                   "/cluewebap/sigirexp/rank_order/original/robust/"+runfile
    qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
               collection+"/qrels/"
    prob_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
               collection+"/background_gain/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
              collection+"/est_eval/"
    if is_log:
        qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+collection\
                   +"/qrels/log_qrels/"
        qrelfile = qrel_dir + str(depth)+"/qrels.input."+gname+".txt"
        out_dir += "fit_eval/leave-out/" + str(depth) + "/"
        prob_dir += "fit_eval/origin/"+str(depth) + "/"
    else:
        qrelfile = qrel_dir + "qrels." + str(depth) + ".txt"
        out_dir += "hybrid_eval/" + str(depth) + "/"
        prob_dir += "hybrid_opt/" + str(depth) + "/"
    dcg_eval = SDCG(k=pd, qrelname=qrelfile, qresname=runfile)
    # dcg_eval.sdcg_eval("testcase/")
    # dcg_eval.sdcg_fit_eval(prob_dir=prob_dir,out_dir=out_dir,is_radom=is_random)
    dcg_eval.sdcg_est_eval(prob_dir=prob_dir, out_dir=out_dir)

if __name__ == "__main__":
    main(sys.argv[1:])

