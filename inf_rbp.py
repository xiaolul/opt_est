from __future__ import division
import numpy as np
from qrel_chk import Qrel
from qres_chk import Qres
import getopt, sys, os

class InfRBP:
    """
    calculate RBP on the sampled
    qrels
    The sampling is in a two strata way
    based on Voorhees paper
    """
    def __init__(self, p, pool_qname, sampled_qname, init_d = 10):
        """
        init evaluation
        :param p: persistence param
        :param qrel_fname: qrel name
        :param init_d: init pooling depth
        """
        self._p = p
        self._pool_qrel = Qrel(pool_qname)
        self._sample_qrel = Qrel(sampled_qname)
        self._init_d = init_d
        self._qid = self._pool_qrel.get_qid()

    def score_estimation(self, run_name, out_dir, dmax = 100):
        """
        based on the sampled documents and number of sampled relevant
        documents to perform the estimation.
        :param run_name:
        :param out_dir:
        :param dmax:
        :return:
        """
        qres = Qres(run_name)
        eval_res = np.zeros(len(self._qid))
        for q in range(0, len(self._qid)):
            curr_pqrel = self._pool_qrel.get_rel_by_qid(self._qid[q])
            curr_sqrel = self._sample_qrel.get_rel_by_qid(self._qid[q])
            curr_qres = qres.get_res_by_id(self._qid[q])
            init_d = self._init_d if len(curr_qres) > self._init_d else len(curr_qres)
            for k in range(0, init_d):
                curr_doc = curr_qres[k].get_doc()
                if curr_doc in curr_pqrel:
                    if curr_pqrel[curr_doc] > 0:
                        eval_res[q] += (1 - self._p) * np.power(self._p, k)
            sampled_score = 0
            sampled_n = 0
            tot_n = 0
            last_pos = dmax if len(curr_qres) > dmax else len(curr_qres)
            for k in range(init_d, last_pos):
                curr_doc = curr_qres[k].get_doc()
                if curr_doc not in curr_pqrel:
                    tot_n += 1
                    if curr_doc in curr_sqrel:
                        sampled_n += 1
                        if curr_sqrel[curr_doc] > 0:
                            sampled_score += (1 - self._p) * np.power(self._p, k)
                elif curr_pqrel[curr_doc] > 0:
                    eval_res[q] += (1 - self._p) * np.power(self._p, k)
            sampled_score = tot_n * ((sampled_score + 0.00001) / (sampled_n + 0.00003))
            eval_res[q] = sampled_score + eval_res[q]
        np.savetxt(fname=out_dir+os.path.basename(run_name)+".rbp", X=eval_res, fmt="%.4f")

    def score_estimation_rm(self, run_name, out_dir, dmax = 100):
        """
        based on the sampled documents and maximum of possible gain.
        which will be similar to the RM mehtod
        :param run_name:
        :param out_dir:
        :param dmax:
        :return:
        """
        qres = Qres(run_name)
        eval_res = np.zeros(len(self._qid))
        for q in range(0, len(self._qid)):
            curr_pqrel = self._pool_qrel.get_rel_by_qid(self._qid[q])
            curr_sqrel = self._sample_qrel.get_rel_by_qid(self._qid[q])
            curr_qres = qres.get_res_by_id(self._qid[q])
            init_d = self._init_d if len(curr_qres) > self._init_d else len(curr_qres)
            for k in range(0, init_d):
                curr_doc = curr_qres[k].get_doc()
                if curr_doc in curr_pqrel:
                    if curr_pqrel[curr_doc] > 0:
                        eval_res[q] += (1 - self._p) * np.power(self._p, k)
            sampled_score = 0
            sampled_n = 0
            tot_n = 0
            last_pos = dmax if len(curr_qres) > dmax else len(curr_qres)
            for k in range(init_d, last_pos):
                curr_doc = curr_qres[k].get_doc()
                if curr_doc not in curr_pqrel:
                    tot_n += (1 - self._p) * np.power(self._p, k)
                    if curr_doc in curr_sqrel:
                        sampled_n += (1 - self._p) * np.power(self._p, k)
                        if curr_sqrel[curr_doc] > 0:
                            sampled_score += (1 - self._p) * np.power(self._p, k)
                elif curr_pqrel[curr_doc] > 0:
                    eval_res[q] += (1 - self._p) * np.power(self._p, k)
            sampled_score = tot_n * ((sampled_score + 0.00001) / (sampled_n + 0.00003))
            eval_res[q] = sampled_score + eval_res[q]
        np.savetxt(fname=out_dir+os.path.basename(run_name)+".rbp", X=eval_res, fmt="%.4f")


def main(argv):
    runfile = ""
    depth = 10
    collection = "robust"
    pd = 100

    try:
        opts, args = getopt.getopt(argv, "r:d:hc:", ["runf", "jfile", "depth"])
    except getopt.GetoptError:
        print('-r <runlist> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-r <runlist> -o <output> -d <depth> -h help')
            sys.exit()
        elif opt in ("-r", "--jfile"):
            runfile = arg
        elif opt in ("-d", "--d"):
            depth = int(arg)
        elif opt in ("-c", "--c"):
            collection = arg

    cstr = collection
    if collection == "tb06":
        cstr = "trec2006"
        pd = 50
    elif collection == "tb05":
        cstr = "trec2005"
        pd = 100
    elif collection == "tb04":
        cstr = "trec2004"
        pd = 80

    if collection == "robust" or collection == "tb06":
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                  "/cluewebap/sigirexp/rank_order/original/" + cstr + "/" + runfile
    else:
        runfile = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                  "/cluewebap/sigirexp/trec_order/original/" + cstr + "/" + runfile

    qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
               collection + "/qrels/"
    prob_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
               collection + "/background_gain/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
              collection + "/est_eval/sample/sample_eval/"+str(depth)+"/"

    pool_name = qrel_dir + "qrels.10.txt"
    sampled_name = qrel_dir + "sample_qrels/qrel."+str(depth)+"-ts.txt-2s"
    inf_rbp = InfRBP(p = 0.95, pool_qname=pool_name, sampled_qname=sampled_name)
    # inf_rbp.score_estimation(out_dir=out_dir, run_name=runfile, dmax = pd)
    inf_rbp.score_estimation_rm(out_dir=out_dir, run_name=runfile, dmax=pd)

if __name__ == "__main__":
    main(sys.argv[1:])
