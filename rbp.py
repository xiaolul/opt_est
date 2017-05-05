from __future__ import division
from qrel_chk import Qrel
from qres_chk import Qres
import sys
import getopt
import os.path
import numpy as np
import futils

class RBP:
    """
    calculate rbp to k
    """

    def __init__(self, p, qresname, qrelname, max_rel = 1,  k=1000):
        """
        init the AP@k calculation
        :param k: evaluation depth, default to the end
        :param qresname: qres file name
        :param qrelname:qrel file name
        """
        self._k = k
        self._qres = Qres(qresname)
        self._qrel = Qrel(qrelname)
        self._qid = self._qrel.get_qid()
        self._default = 0.0
        self._p = p
        self._rname = os.path.basename(qresname)
        self._gname = os.path.basename(qrelname).split(".")[2]
        self._qrel_dir = os.path.dirname(qrelname)
        self._qrel_dir = os.path.dirname(self._qrel_dir) + "/"
        self._binary = True if max_rel == 1 else False
        self._max_rel = max_rel

    def rbp_eval(self, out_dir):
        """
        evaluate rbp, based on the input qrel file only
        :param out_dir: output dir
        :return:
        """
        score_str = [str(self._k) + "," + "p="+str(self._p) + ","] * len(self._qid)
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
                        rel_val = 1 if self._binary else curr_rel[curr_doc]
                        curr_score += rel_val * np.power(self._p, i)
                else:  # the residual part.
                    residual += self._max_rel * np.power(self._p, i)
            for i in range(last_pos, 1000):
                residual += self._max_rel * np.power(self._p, i)
            score_str[q] = str(self._qid[q]) + "," + score_str[q] + \
                           "s= {:.4f},r={:.4f}".format((1-self._p) * curr_score,
                                                       (1- self._p) * residual).strip()
        with open(out_dir + self._rname + ".rbp", "w") as fout:
            for s_str in score_str:
                fout.write(s_str + "\n")

    def rbp_eval_vary_p(self, plist, out_dir):
        """
        evaluate rbp, with different p-values.
        :param plist: list of p values. by default it is 0.5,0.6,0.7,0.8,0.95
        :param out_dir:
        :return:
        """
        pval = np.array(plist)
        score = np.zeros((len(self._qid), len(plist)))
        residual = np.zeros((len(self._qid), len(plist)))
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            last_pos = self._k if len(curr_res) > self._k else len(curr_res)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = 1 if self._binary else curr_rel[curr_doc]
                        score[q, :] += np.power(pval, i) * rel_val
                else:
                    residual[q, :] += np.power(pval, i) * self._max_rel
            for i in range(last_pos, 1000):
                residual[q,:] += np.power(pval, i) * self._max_rel
        s_name = out_dir + self._rname + ".rbp"   # base score fname
        r_name = out_dir + self._rname + "-residual.rbp"  # residual fname
        np.savetxt(fname=s_name, X=(1-pval) * score, fmt="%.4f", delimiter=",")
        np.savetxt(fname=r_name, X=(1 - pval) * residual, fmt="%.4f", delimiter=",")


    def rbp_fit_eval(self, prob_dir, out_dir, is_radom=False):
        """
        calculate rbp score based on fitted background gain
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
            prob_mat = np.loadtxt(fname,delimiter=" ",dtype=float)
            m_num = prob_mat.shape[1]
            if score.shape[1] > m_num + 3:
                score = score[:, (m_num + 3)]
            last_pos = len(curr_res) if len(curr_res) < self._k else self._k
            final_pos = 1000 if len(curr_res) > 1000 else len(curr_res)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    rel = curr_rel[curr_doc]
                    if rel > 0:
                        rel = 1 if self._binary else rel
                        score[q,:] += rel * np.power(self._p, i)
                else:   # the document is not judged
                    score[q, -1] += self._max_rel * np.power(self._p, i)
                    score[q, 1:(m_num+1)] += np.power(self._p, i) * prob_mat[i,:]
            for i in range(last_pos, final_pos):
                score[q, -1] += self._max_rel *  np.power(self._p, i)
                score[q, 1:(m_num + 1)] += np.power(self._p, i) * prob_mat[i, :]
            score[q, :] *= (1 - self._p)
            # calculate the RM method
            delta = score[q, -1] - score[q, 0]
            if delta == 1:
                score[q, -2] = score[q, 0] + delta * 0.01
            else:
                score[q, -2] = score[q, 0] + delta * score[q, 0] / (1 - delta)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.rbp"
        else:
            fname += ".rbp"
        np.savetxt(fname=fname, X=score, fmt="%.4f",delimiter=",")

    def rbp_fit_eval_threashold(self, prob_dir, out_dir, gamma_file, is_radom=False):
        """
        evaluate the rbp value, but adjust per query based on gamma
        :param prob_dir:
        :param out_dir:
        :param gamma_file:
        :param rnum: run numbers, hard code for now, but pass parameter later
        :param is_radom:
        :return:
        """
        score = np.zeros((len(self._qid), 8))
        gamma_mat = np.loadtxt(gamma_file,delimiter=",",dtype=float)
        gamma_mat = gamma_mat[:,3]
        threshold = 0.018
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            fname = prob_dir + str(self._qid[q])
            if is_radom:
                fname += "-r.txt"
            else:
                fname += ".txt"
            prob_mat = np.loadtxt(fname,delimiter=" ",dtype=float)
            m_num = prob_mat.shape[1]
            if score.shape[1] > m_num + 3:
                score = score[:, (m_num + 3)]
            last_pos = len(curr_res) if len(curr_res) < self._k else self._k
            final_pos = 1000 if len(curr_res) > 1000 else len(curr_res)
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    rel = curr_rel[curr_doc]
                    if rel > 0:
                        rel = 1 if self._binary else rel
                        score[q,:] += rel * np.power(self._p, i)
                else:   # the document is not judged
                    score[q, -1] += self._max_rel * np.power(self._p, i)
                    if gamma_mat[q] > threshold:
                        score[q, 1:(m_num+1)] += np.power(self._p, i) * prob_mat[i,:]
            for i in range(last_pos, final_pos):
                score[q, -1] += self._max_rel * np.power(self._p, i)
                if gamma_mat[q] > threshold:
                    score[q, 1:(m_num + 1)] += np.power(self._p, i) * prob_mat[i, :]
            score[q, :] *= (1 - self._p)
            # calculate the RM method
            delta = score[q, -1] - score[q, 0]
            if delta == 1:
                score[q, -2] = score[q, 0] + delta * 0.01
            else:
                score[q, -2] = score[q, 0] + delta * score[q, 0] / (1 - delta)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-gamma-r.rbp"
        else:
            fname += "-gammar.rbp"
        np.savetxt(fname=fname, X=score, fmt="%.4f",delimiter=",")

    def rbp_est_eval(self, prob_dir, out_dir, is_radom=False):
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
                        rel_val = 1 if self._binary else curr_rel[curr_doc]
                        score[q, 0:init_m] += np.power(self._p, i) * rel_val
                else:  # use estimated values instead
                    est_rel = np.array(prob_mat[curr_doc])
                    score[q, 0:init_m] += np.power(self._p, i) * est_rel
            final_pos = 1000 if len(curr_res) > 1000 else len(curr_res)
            for i in range(last_pos, final_pos):
                curr_doc = curr_res[i].get_doc()
                est_rel = np.zeros(4)
                if curr_doc not in prob_mat:
                    print (q, curr_doc)
                else:
                    est_rel = np.array(prob_mat[curr_doc])
                score[q, 0:init_m] += np.power(self._p, i) * est_rel
            score[q, 0:init_m] *= (1 - self._p)
        fname = out_dir + self._rname
        if is_radom:
            fname += "-r.rbp"
        else:
            fname += ".rbp"
        np.savetxt(fname=fname, X=score[:, 0:init_m], fmt="%.4f", delimiter=",")

    def rbp_est_eval_threshold(self, prob_dir,
                               out_dir,
                               gamma_file,
                               fit_dir="",
                               is_random=False):
        """
        evaluate score based on the estimated gain.
        per document, if the document has not been estimated,
        use the fitted gain instead.
        Notice that the residual will not be calculated in this case,
        everything is based on the estimated only.
        :param prob_dir: per document background gain dir
        :param out_dir: output dir
        :param gamma_file: calculated gamma
        :param fit_dir: for use of leave one out.
        :param is_random: use randomized result?
        :return:
        """
        score = np.zeros((len(self._qid), 4))
        init_m = score.shape[1]
        gamma_mat = np.loadtxt(gamma_file, delimiter=",", dtype=float)
        gamma_mat = gamma_mat[:, 3]
        threshold = 0.018
        # threshold = 0
        param_mat = np.loadtxt(prob_dir+"param.txt",dtype=float,delimiter=",")[:,1:]
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            curr_res = self._qres.get_res_by_id(self._qid[q])
            prob_name = prob_dir + str(self._qid[q])
            fit_name = fit_dir + str(self._qid[q])
            if is_random:
                prob_name += "-r.txt"
                fit_name += "-r.txt"
            else:
                prob_name += "-prob.txt"
                fit_name+= ".txt"
            prob_mat, mnum = futils.read_csv_to_dict(prob_name, is_prob=True)
            if init_m > mnum:   # shrink to fit
                init_m = mnum
            last_pos = self._k if len(curr_res) > self._k else len(curr_res)
            # only three methods are considered, and only the last one is considered
            fit_val = np.loadtxt(fit_name, dtype=float)[-1, 2:5]
            missing_val = np.zeros(init_m)
            cnt = 0
            for i in range(0, init_m):
                if i == 0 or 1:
                    missing_val[i] = np.dot(param_mat[q,cnt : cnt+3], fit_val)
                else:
                    tmp = fit_val
                    tmp[tmp == 0] = 10 ** -6
                    missing_val[i] = np.exp(np.dot(param_mat[q,cnt : cnt+3], np.log(missing_val)))
                cnt += 3
            for i in range(0, last_pos):
                curr_doc = curr_res[i].get_doc()
                if curr_doc in curr_rel:
                    if curr_rel[curr_doc] > 0:
                        rel_val = 1 if self._binary else curr_rel[curr_doc]
                        score[q, 0:init_m] += np.power(self._p, i) * rel_val
                else:  # use estimated values instead
                    if gamma_mat[q] > threshold:
                        if curr_doc in prob_mat:
                            est_rel = np.array(prob_mat[curr_doc])
                        else:
                            est_rel = missing_val
                        score[q, 0:init_m] += np.power(self._p, i) * est_rel
            final_pos = 1000 if len(curr_res) > 1000 else len(curr_res)
            for i in range(last_pos, final_pos):
                curr_doc = curr_res[i].get_doc()
                if gamma_mat[q] > threshold:
                    if curr_doc in prob_mat:
                        est_rel = np.array(prob_mat[curr_doc])
                    else:
                        est_rel = missing_val
                    score[q, 0:init_m] += np.power(self._p, i) * est_rel
            score[q, 0:init_m] *= (1 - self._p)
        fname = out_dir + self._rname
        if is_random:
            fname += "-gamma-r.rbp"
        else:
            fname += ".rbp"
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
    is_vary = False
    is_fit = False
    is_gamma = False
    fit_dir = ""
    try:
        opts, args = getopt.getopt(argv, "r:j:d:c:g:m:tsh", ["runf", "jfile", "depth"])
    except getopt.GetoptError:
        print('-r <runlist> -j <qrelfile> -d <depth> -c <collection> -g <gname> -s shuffle -t threshold -h help')
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
        elif opt in ("-c", "--collection"):
            collection = arg
        elif opt in ("-g", "--grp name"):
            gname = arg
            is_log = True
        elif opt in ("-s", "--shuffle"):
            is_random = True
        elif opt in ("-t", "--threshold"):
            is_gamma = True
        elif opt in ("-m", "--vary_p"):
            m = int(arg)
            if m == 0:
                is_plain = True
            elif m == 1:
                is_vary = True
                is_plain = False
            elif m == 2:
                is_fit = True
                is_plain = False
            elif m == 3:
                is_fit = False
                is_plain = False
    p = 0.95
    if collection == "tb06":
        pd = 50
    elif collection == "tb04":
        pd = 80
    elif collection == "2010":
        pd = 20
    cstr = collection
    if collection == "tb06":
        cstr = "trec2006"
    elif collection == "tb05":
        cstr = "trec2005"
    elif collection == "tb04":
        cstr = "trec2004"

    if collection == "robust" or collection == "tb06":
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                   "/cluewebap/sigirexp/rank_order/original/"+cstr + "/" + runfile
    else:
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                  "/cluewebap/sigirexp/trec_order/original/" + cstr + "/" + runfile

    qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
               collection+"/qrels/"
    prob_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
               collection+"/background_gain/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
              collection+"/est_eval/"
    gamma_file = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
                 "naive_estimator/"+collection+"/"+str(depth)+"-rsim.txt"

    if is_log:
        qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + collection \
                   + "/qrels/log_qrels/"
        qrelfile = qrel_dir + str(depth) + "/qrels.input." + gname + ".txt"
        gamma_file = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
                 "naive_estimator/"+collection+"/"+gname+"-"+str(depth)+"-rsim.txt"
        if is_fit:
            out_dir += "fit_eval/leave-out/" + str(depth) + "/"
            prob_dir += "fit/leave-out/"+gname+"/" +str(depth) + "/"
        else:
            out_dir += "log_hybrid_eval/" + str(depth) + "/"
            fit_dir = prob_dir + "fit/leave-out/"+gname+"/" +str(depth) + "/"
            prob_dir += "leave-out/hybrid_opt/" + gname + "/" + str(depth) + "/"
    elif is_fit:
        qrelfile = qrel_dir + "qrels."+str(depth)+".txt"
        prob_dir += "fit/origin/" + str(depth) + "/"
        out_dir += "fit_eval/origin/" + str(depth) + "/"
    elif not is_plain:
        qrelfile = qrel_dir + "sample_qrels/qrel." + str(depth) + "-ts.txt"
        out_dir += "sample/inf_hybrid/" + str(depth) + "/"
        fit_dir = prob_dir +  "fit/origin/" + str(10) + "/"
        prob_dir += "sample_rbp/hybrid_opt/" + str(depth) + "/"
    if is_vary:  # for varying p
        out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
                  "rbp_score/" + collection + "/" + str(depth) + "/"
    rbp_eval = RBP(p=p, k=pd, qrelname=qrelfile, qresname=runfile)
    plist = [0.5,0.6,0.7,0.8,0.95]
    if is_fit:
        if is_gamma:
            rbp_eval.rbp_fit_eval_threashold(prob_dir,out_dir,gamma_file,is_random)
        else:
            rbp_eval.rbp_fit_eval(prob_dir, out_dir, is_random)
    else:
        if is_plain:
            rbp_eval.rbp_eval_vary_p(plist, out_dir)
        else:
            if is_gamma:
                rbp_eval.rbp_est_eval_threshold(prob_dir=prob_dir,
                                                out_dir=out_dir,
                                                gamma_file=gamma_file,
                                                fit_dir=fit_dir)
            else:
                rbp_eval.rbp_est_eval(prob_dir=prob_dir,
                                      out_dir=out_dir)

if __name__ == "__main__":
        main(sys.argv[1:])
