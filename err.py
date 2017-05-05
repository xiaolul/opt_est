from __future__ import division
from qrel_chk import Qrel
from qres_chk import Qres
import os
import futils
import numpy as np
import sys
import getopt

class ERR:
    """
    evaluation using ERR metric
    with residual by default.
    """
    def __init__(self, qrelname, qresname, k=1000, max_rel = 1):
        """
        initialize calculation of calculating ERR
        :param qrelname: name of qrel
        :param qresname: name of qres
        :param k: the evaluation depth
        :param max_rel: maximum relevance. 1 is binary
        """
        self._binary = True if max_rel==1 else False
        self._gmax = 2 ** max_rel
        self._k = k
        self._qrel = Qrel(qrelname)
        self._qres = Qres(qresname)
        self._qid = self._qrel.get_qid()
        self._max_rel = self._rel_convert(max_rel)
        self._rname = os.path.basename(qresname)
        self._gname = os.path.basename(qrelname).split(".")[2]
        self._qrel_dir = os.path.dirname(qrelname) + "/"

    def _rel_convert(self, rel):
        """
        convert the relevance based on the
        gain
        :return: mapped value
        """
        return (np.power(2, rel) -1 )/ self._gmax

    def err_eval(self, out_dir):
        """
        calculate ERR based on the qrel only
        :param out_dir: output dir
        :return: dump out the score
        """
        score_str = [str(self._k) + "," + "rmax=" + str(self._max_rel) + ","] * len(self._qid)
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            last_pos = self._k if len(curr_res) > self._k else len(curr_res)
            score = 0
            residual = 0
            p = 1
            p_res = 1
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = self._rel_convert(1) if self._binary \
                                  else self._rel_convert(curr_rel[curr_doc])
                        score += p * rel_val/(i+1)
                        p *= (1 - rel_val)
                        p_res *= (1 - rel_val)
                else:
                    residual += p_res * self._max_rel / (i + 1)
                    p_res *= (1 - self._max_rel)
            for i in range(last_pos, 1000):
                residual += p_res * self._max_rel / (i + 1)
                p_res *= (1 - self._max_rel)
            score_str[q] = str(self._qid[q]) + "," + score_str[q] + \
                           "s={:.4f},r={:.4f}".format(score, residual).strip()
        with open(out_dir + self._rname + ".err", "w") as fout:
            for s_str in score_str:
                fout.write(s_str + "\n")

    def err_fit_eval(self,prob_dir, out_dir, is_radom=False):
        """
        calculate ERR score based on the fitting method and the
        RM method
        :param prob_dir: fitting prob dir
        :param out_dir: ouput dir
        :param is_radom: use randomized fitting?
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
            p = np.ones(m_num + 3)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = self._rel_convert(1) \
                            if self._binary \
                            else self._rel_convert(curr_rel[curr_doc])
                        score[q, :] += p * rel_val/(i+1)
                        p *= (1 - rel_val)
                else:  # the document is not judged
                    fit_val = self._rel_convert(prob_mat[i, :])
                    score[q, -1] += p[-1] * self._max_rel/(i+1)
                    score[q, 1:(m_num + 1)] += p[1:(m_num + 1)] * fit_val/ (i+1)
                    p[1:(m_num + 1)] *= (1 - fit_val)
                    p[-1] *= (1- self._max_rel)
            for i in range(last_pos, 1000):
                fit_val = self._rel_convert(prob_mat[i, :])
                score[q, -1] += p[-1] * self._max_rel / (i + 1)
                score[q, 1:(m_num + 1)] += p[1:(m_num + 1)] * fit_val / (i + 1)
                p[1:(m_num + 1)] *= (1 - fit_val)
                p[-1] *= (1 - self._max_rel)
            # calculate the RM method
            delta = score[q, -1] - score[q, 0]
            if delta == 1:
                score[q, -2] = score[q, 0] + delta * 0.01
            else:
                score[q, -2] = score[q, 0] + delta * score[q, 0] / (1 - delta)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.err"
        else:
            fname += ".err"
        np.savetxt(fname=fname, X=score, fmt="%.4f", delimiter=",")

    def err_est_eval(self, prob_dir, out_dir, is_radom=False):
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
            p = np.ones(mnum)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = self._rel_convert(1) if self._binary \
                            else self._rel_convert(curr_rel[curr_doc])
                        score[q, 0:init_m] += p * rel_val / (i + 1)
                        p *= (1 - rel_val)
                else:  # use estimated values instead
                    est_rel = self._rel_convert(np.array(prob_mat[curr_doc]))
                    score[q, 0:init_m] += p * est_rel / (i + 1)
                    p *= (1 - est_rel)
            for i in range(last_pos, 1000):
                curr_doc = curr_res[i].get_doc()
                est_rel = self._rel_convert(np.array(prob_mat[curr_doc]))
                score[q, 0:init_m] += p * est_rel / (i + 1)
                p *= (1 - est_rel)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.err"
        else:
            fname += ".err"
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
               collection+"/background_gain/fit/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
              collection+"/est_eval/fit_eval/"
    if is_log:
        qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+collection\
                   +"/qrels/log_qrels/"
        qrelfile = qrel_dir + str(depth)+"/qrels.input."+gname+".txt"
        out_dir += "leave-out/" + str(depth) + "/"
        prob_dir += "origin/"+str(depth) + "/"
    else:
        qrelfile = qrel_dir + "qrels."+str(depth)+".txt"
        out_dir += "origin/"+str(depth) + "/"
        prob_dir += "origin/" + gname + "/" + str(depth) + "/"
    err_eval = ERR(k=pd, qrelname=qrelfile, qresname=runfile)
    # err_eval.err_eval("testcase/")
    err_eval.err_fit_eval(prob_dir=prob_dir,out_dir=out_dir,is_radom=is_random)

if __name__ == "__main__":
    main(sys.argv[1:])



