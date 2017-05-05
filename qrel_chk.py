from qres_chk import Qres,ResTpl
import sys
from collections import defaultdict


class QrelTpl:
    """
    Class of qrel tuple. An example of Qrel tuple should be
    301 0 FBIS3-10082 1
    The qrel information are not allowed to be modified.
    """

    def __init__(self, qrel_str):
        """
        init a qrel tuple, split the input string by any white space
        :param qrel_str:
        :return:
        """
        qrel_arr = qrel_str.strip().split()
        if len(qrel_arr) < 4:
            print >> sys.stderr, 'From QrelTuple: Error in format'
        self._qid = int(qrel_arr[0])
        self._doc = qrel_arr[2]
        self._rel = float(qrel_arr[3])

    def get_qid(self):
        """
        get qid.
        :return:qid
        """
        return int(self._qid)

    def get_doc(self):
        """
        get document id
        :return:
        """
        return self._doc

    def get_rel(self):
        if self._rel > 0:
            return self._rel
        else:
            return 0

    def get_str(self):
        return (self._qid + " 0 " + self._doc + " " + self._rel).strip()


class Qrel:
    """
    Read the Qrel from file and form qrel instance
    """

    def __init__(self, fname):
        self._judged = defaultdict(set)
        self._qrels = defaultdict(dict)
        self._totqrel = {}
        with open(fname, "r") as fin:
            for line in fin:
                if len(line.strip()) > 1:
                    tpl = QrelTpl(line)
                    self._judged[tpl.get_qid()].add(tpl.get_doc())
                    self._qrels[tpl.get_qid()][tpl.get_doc()] = tpl.get_rel()
                    if tpl.get_qid() not in self._totqrel:
                        self._totqrel[tpl.get_qid()] = 0
                    if tpl.get_rel() > 0:
                        self._totqrel[tpl.get_qid()] += min(tpl.get_rel(), 1.0)
        fin.close()

    def get_judged_by_qid(self, qid):
        return self._judged[qid]

    def get_rel_by_qid(self,qid):
        """
        get relevance ny qid
        :param qid:
        :return:a dictionary, key is the document
        """
        if self._qrels.has_key(qid):
            return self._qrels[qid]
        else:
            return -1

    def get_qid(self):
        """
        :return list of qids. query that are pooled.
        """
        return self._qrels.keys()

    def tot_rel(self, qid):
        return self._totqrel[qid]


def condense_run(qrel,run):
    id_list = run.get_qid_set()
    for id in id_list:
        curr_run = run.get_res_by_id(id)
        curr_qrel = qrel.get_judged_by_qid(id)
        for res_tpl in curr_run:
            if res_tpl.get_doc() in curr_qrel:
                print res_tpl.get_str()

def match_qrel(qrel,run):
    """
    match qres with qrel, and generate a sequence
    :param qrel:
    :param run:
    :return:
    """
    id_list = run.get_qid_set()
    for qid in id_list:
        curr_qres = run.get_res_by_id(qid)
        curr_qrel = qrel.get_rel_by_qid(qid)
        with open(str(qid)+"-seq.txt","a") as fout:
            for res_tpl in curr_qres:
                if curr_qrel.has_key(res_tpl.get_doc()):
                    fout.write(curr_qrel[res_tpl.get_doc()]+"\n")
                else:
                    #The unjudged case
                    fout.write("-1"+"\n")
        fout.close()


if __name__ == "__main__":
    # use command line as input
    run_name = sys.argv[1]
    rel_name = sys.argv[2]
    run = Qres(run_name)
    qrel = Qrel(rel_name)
    match_qrel(qrel,run)
    # condense_run(qrel,run)