from __future__ import division
import sys
import getopt
import numpy as np
from os import listdir
from os.path import isfile, join


class GofTest:
    """
    This is used for testing purely the goodness-of-fit
    for those rank-level estimators.
    """
    def __init__(self, ref_dir, suf='-rel.txt', delim=','):
        """
        load the reference name
        :param ref_dir: reference directory
        :param suf: file suffix
        :param delim: delimeter of current file
        """
        self._fname = sorted([int(f.replace(suf,'')) for f in listdir(ref_dir) if isfile(join(ref_dir, f))])
        self._suf = suf
        self._delim = delim
        self._tnum = len(self._fname)
        if len(self._fname) == 0:
            raise ValueError('No file can be found')
        else:
            tmp_rel = np.loadtxt(ref_dir + str(self._fname[0]) + suf, dtype=int, delimiter=delim)
            (self._d, self._n)  =  np.shape(tmp_rel)
            self._empirical_gain = np.zeros((self._d, self._tnum))
            for f in range(0, self._tnum):
                curr_rank_rel = np.loadtxt(ref_dir + str(self._fname[f]) + suf, delimiter=delim)
                self._empirical_gain[:, f] = np.mean(curr_rank_rel, axis=1)


    def compute_gof_rmse(self, dir_str, suf, delim, pd, m=5):
        """
        compute gof rmse values
        :param dir_str: directory
        :param suf: file suffix
        :param delim: delimeter
        :param pd: pooling depth
        :param m: total number of rank-level estimators
        :return:
        """
        gof_rmse = np.zeros((self._tnum, m))
        ref_d = self._empirical_gain.shape[0]
        for i in range(0, self._empirical_gain.shape[1]):
            estimate = np.loadtxt(dir_str+str(pd)+"/"+str(self._fname[i])+suf, delimiter=delim)[0:ref_d,:]
            tmp_delta = np.zeros(np.shape(estimate))
            for j in range(0, m):
                tmp_delta[:, j] = np.square(estimate[:, j] - self._empirical_gain[:, i])
            gof_rmse[i, :] = np.sqrt(np.mean(tmp_delta[0:pd, :], axis=0))
        return np.mean(gof_rmse, axis=0)




def main(argv):
    ref_dir = "" # relevance matrix from rank 1 to d
    dir_str = "" # rank-level estimations.
    eval_tool = GofTest(ref_dir=ref_dir)
    for i in xrange(10, 60, 10):
        mean_rmse = eval_tool.compute_gof_rmse(dir_str=dir_str, suf=".txt", delim=" ", pd=i)
        np.set_printoptions(precision=4)
        print mean_rmse



if __name__ == "__main__":
    main(sys.argv[1:])
