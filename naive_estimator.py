from __future__ import division
from qrel_chk import Qrel
import futils
import numpy as np
import getopt
import sys
import os.path

class NaiveEstimator:
    """
    calculate sample coverage.
    Try to estimate total number of
    relevant documents using Chao92
    """

    def __init__(self, qrelname, d, gname, gidx):
        """
        init Chao92 estimator
        :param qrelname: qrel name
        :param d: considered pooling depth
        """
        self._d = d
        self._qrel = Qrel(qrelname)
        self._qid = self._qrel.get_qid()
        self._isout = False
        if len(gidx) > 0:
            self._gname = gname
            self._gidx = np.array(gidx)
            self._isout = True

    def naive_estimator(self, rank_dir, out_dir):
        jaccard_mat = np.zeros((len(self._qid), 6))
        jaccard_mat[: ,0] = self._qid
        col_num = 0
        for q in range(0, len(self._qid)):
            curr_rel = self._qrel.get_rel_by_qid(self._qid[q])
            doc_idx = []
            col_num = 0
            with open(rank_dir + str(self._qid[q]) + "-rank.txt", "rb") as fin:
                i = 0
                for lines in fin:
                    curr_line = lines.strip().split(",")
                    if col_num == 0:
                        col_num = len(curr_line)
                    if curr_line[0] in curr_rel:
                        if curr_rel[curr_line[0]] > 0:
                            doc_idx.append(i)
                    i += 1
            if len(doc_idx) == 0:
                jaccard_mat[q, 1:] = 0
            else:
                rank_mat = np.loadtxt(rank_dir + str(self._qid[q]) + "-rank.txt",
                                      usecols=range(1, col_num),
                                      delimiter=",", dtype=int)
                rank_mat = rank_mat[np.array(doc_idx), :]
                rank_mat[rank_mat >= self._d] = -2
                rank_mat[rank_mat >= 0 ] = 1
                rank_mat[rank_mat < 0] = 0
                col_num -= 1
                k = len(doc_idx)
                for j in range(0, (col_num - 1)):
                    sys_i = rank_mat[:, j]
                    for r in range(j + 1, col_num):
                        sys_j = rank_mat[:, r]
                        if sum(sys_i | sys_j) > 0:
                            jaccard_mat[q, 1] += sum(sys_i & sys_j) / sum(sys_i | sys_j)
                        else:
                            jaccard_mat[q, 1] += 0
                f_stat = np.sum(rank_mat, axis=1)
                uniq_f_stat = np.unique(f_stat)
                tot_sample = np.sum(rank_mat)
                for f in uniq_f_stat:
                    jaccard_mat[q, 3] += f * (f - 1) * len(f_stat[f_stat == f])
                jaccard_mat[q, 2] = 1 - (len(f_stat[f_stat == 1]) / tot_sample)
                if jaccard_mat[q, 2] == 0:
                    jaccard_mat[q, 3] = 0
                    jaccard_mat[q, 4] = k
                else:
                    tmp_val = (k / jaccard_mat[q, 2]) / ((tot_sample - 1) * tot_sample)
                    gamma_sq = max(jaccard_mat[q, 3]*tmp_val - 1, 0)
                    jaccard_mat[q, -2] = (k / jaccard_mat[q, 2]) + \
                                         gamma_sq * (tot_sample*(1-jaccard_mat[q, 2]))/jaccard_mat[q, 2]
                    jaccard_mat[q, 3] = np.sqrt(gamma_sq)/col_num
                jaccard_mat[q, -1] = k
        jaccard_mat[:, 1] /= ((col_num - 1) * col_num * 0.5)
        jaccard_mat[:, 2] /= col_num  # average to per run
        fname = out_dir+str(self._d)+"-rsim.txt"
        if self._isout:
            fname = out_dir + self._gname + "-" + str(self._d) + "-rsim.txt"
        np.savetxt(fname=fname,
                   X=jaccard_mat, fmt="%.4f",delimiter=",",
                   header="qid, jaccard, hat_c, gamma, est_N, tot_N")



    def avg_sim_all(self, rank_dir, out_dir, k= 1000):
        """
        Calculate different similarity measurement
        :param rank_dir:
        :param out_dir:
        :param k: considered depth
        :return:
        """
        jaccard_mat = np.zeros((len(self._qid), 4))
        col_num = 0
        jaccard_mat[:, 0] = self._qid
        for i in range(0, len(self._qid)):
            with open(rank_dir+str(self._qid[i])+"-rank.txt", "rb") as fin:
                col_num = len(fin.readline().strip().split(","))
            rank_mat = np.loadtxt(rank_dir+str(self._qid[i])+"-rank.txt",
                                  usecols=range(1, col_num),
                                  delimiter=",", dtype=int)
            if self._isout:
                rank_mat = np.delete(rank_mat,self._gidx, axis=1)
            rank_mat[rank_mat >= k] = -2
            rank_mat[rank_mat > -1] = 1
            rank_mat[rank_mat < 0] = 0
            col_num = rank_mat.shape[1]
            for j in range(0, (col_num-1)):
                sys_i = rank_mat[:, j]
                for r in range(j+1, col_num):
                    sys_j = rank_mat[:, r]
                    jaccard_mat[i,1] += sum(sys_i & sys_j) / sum(sys_i | sys_j)
            f_stat  = np.sum(rank_mat, axis=1)
            uniq_f_stat = np.unique(f_stat)
            tot_sample = np.sum(rank_mat)
            for f in uniq_f_stat:
                jaccard_mat[i, -1] += f* (f-1) * len(f_stat[f_stat==f])
            jaccard_mat[i, 2] = 1 - (len(f_stat[f_stat == 1])/tot_sample)
            tmp_val = (rank_mat.shape[0]/jaccard_mat[i, 2])/((tot_sample - 1)*tot_sample)
            jaccard_mat[i, -1] *= tmp_val
            jaccard_mat[i, -1] = np.sqrt(max(jaccard_mat[i, -1]-1, 0)) / col_num
        jaccard_mat[:, 1] /= ((col_num-1) * col_num * 0.5)
        jaccard_mat[:, 2:] /= col_num  # average to per run
        fname = out_dir + str(self._d) + "-asim.txt"
        if self._isout:
            fname = out_dir + self._gname + "-" + str(self._d) + "-asim.txt"
        print  fname
        np.savetxt(fname=fname,
                   X=jaccard_mat, fmt="%.4f",delimiter=",",
                   header="qid, jaccard, hat_c, gamma")


def main(argv):
    collection = "robust"
    depth = 10
    qrelname = ""
    gname=""
    try:
        opts, args = getopt.getopt(argv, "d:c:g:h", ["jfile", "depth", "collection"])
    except getopt.GetoptError:
        print('-j <qrelfile> -d <depth> -c <collection> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-j <qrelfile> -d <depth> -c <collection> -h help')
            sys.exit()
        # elif opt in ("-j", "--jfile"):
        #     qrelname = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-c", "--depth"):
            collection = arg
        elif opt in ("-g", "--depth"):
            gname = arg

    rank_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
               collection + "/doc_rank/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/naive_estimator/" + \
              collection + "/"
    gfile = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
            collection+"/qrels/log_qrels/input."+gname+".txt"

    gidx = []
    if gname!="":
        with open(gfile, "rb") as fin:
            for lines in fin:
                gidx.append(int(lines.strip()))

    if len(gidx) == 0:
        qrelname = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
                   collection+"/qrels/qrels." + str(depth) + ".txt"
    else:
        qrelname = "/research/remote/petabyte/users/xiaolul/pool_probability/" + \
                   collection + "/qrels/log_qrels/" + str(depth) + \
                   "/qrels.input."+gname+".txt"
    print  gname
    naive_est = NaiveEstimator(qrelname=qrelname, d=depth,
                               gname = gname, gidx = gidx)
    naive_est.naive_estimator(rank_dir=rank_dir, out_dir=out_dir)
    naive_est.avg_sim_all(rank_dir=rank_dir, out_dir= out_dir, k =depth)
    if not os.path.isfile(out_dir+"1000-asim.txt") and len(gidx) == 0:
        naive_est.avg_sim_all(rank_dir=rank_dir, out_dir=out_dir, k=1000)

if __name__ == "__main__":
    main(sys.argv[1:])