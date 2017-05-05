from __future__ import division
import csv
from collections import defaultdict
from qres_chk import Qres
from qrel_chk import Qrel

class DumpFeature:
    """
    dump feature
    for the GBR method
    """
    def __init__(self, qrel, rank_dir, pd, p):
        """
        init class
        :param qrel: Qrel
        :param rank_dir: dir of doc-rank
        :param pd: pooling depth, the original one.
        :param p: rbp parameter
        """
        self._qrel = qrel
        self._rank_dir = rank_dir
        self._qid = self._qrel.get_qid()
        self._pd = pd
        self._p  = p

    def _read_rank_mat(self, fname):
        """
        read the document-rank matrix in to dictionary-list
        :param fname: path of the doc-rank file
        :return: default dict with document as key and rank list as value
        """
        doc_mat = defaultdict(list)
        with open(fname, 'rb') as fin:
            reader = csv.reader(fin, delimiter=",")
            for row in reader:
                doc_mat[row[0]] = [int(x) for x in row[1:]]
        return doc_mat


    def get_feature(self, is_random=False):
        """
        extracting features for the GBR method
        rank of 1st rel@10, rank of first unrel@10, rank of first unj@10
        ...
        rank of 1st rel@100, rank of first unrel@100, rank of first unj@100
        residual@current pooling step
        rbp_base@current pooling step
        :return:
        """
        for q in self._qid:
            curr_rel = self._qrel.get_rel_by_qid(q)
            doc_mat = self._read_rank_mat(self._rank_dir+str(q)+"-rank.txt")



