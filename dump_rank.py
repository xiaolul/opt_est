from qres_chk import Qres
from qrel_chk import Qrel
import sys
import getopt
import numpy as np

class DumpRank:
    """
    dump rank from a given list of runs and
    a list of qrels.
    please using full qrels
    """
    def __init__(self, runlist, qrels):
        """
        ctr
        :param runlist: list of qres
        :param qrels: list of qrels.
        """
        self._runlist = runlist
        self._qrels = qrels
        self._topic = self._qrels.get_qid()
        self._runnum = len(self._runlist)

    def dump_rank(self,out_pref):
        """
        dump ranks based on runlist and qrels
        if the document is not in the runlist,
        then use ``-1'' as the value, since
        it is the last position in python by default
        :param out_pref:prefix of output dir
        :return: null
        """
        for t in self._topic:
            doc_list = self._qrels.get_judged_by_qid(t)
            rank_mat = {d: [-1]*self._runnum for d in doc_list}
            for i in range(0, self._runnum):  #iterating through all runs
                curr_rank = self._runlist[i].get_res_by_id(t)
                for k in range(0, len(curr_rank)):  # iterating through rank list
                    curr_doc = curr_rank[k].get_doc()
                    if curr_doc in rank_mat:
                        rank_mat[curr_doc][i] = k
                    else:
                        rank_mat[curr_doc] = [-1]*self._runnum
                        rank_mat[curr_doc][i] = k
            out_dir = out_pref + str(t) + "-rank.txt"
            with open(out_dir,'w') as fout:
                for doc, ranklist in rank_mat.iteritems():
                    rank_str = doc + "," + ','.join(map(str,ranklist))
                    fout.write(rank_str.strip()+"\n")

    def dump_rel(self, out_pref, pd):
        """
        dump rel matrix to pooling depth
        no document is needed, only matrix of 0-1
        by default, only binary is conisidered
        :param out_pref: predix of output dir
        :param pd: pooling depth
        :return:
        """
        for t in self._topic:
            curr_qrel = self._qrels.get_rel_by_qid(t)
            rel_mat = np.zeros((pd,self._runnum),dtype=int)  #  init d * runnum mat with zero
            for i in range(0, self._runnum):
                curr_rank = self._runlist[i].get_res_by_id(t)
                last_pos = pd if len(curr_rank) > pd else len(curr_rank)
                for k in range(0, last_pos):  # iterating through rank list
                    curr_doc = curr_rank[k].get_doc()
                    if curr_doc in curr_qrel:
                        if curr_qrel[curr_doc] > 0:
                            rel_mat[k,i] = 1
            out_dir = out_pref+str(t) + "-rel.txt"
            np.savetxt(fname=out_dir,X=rel_mat,delimiter=",",fmt="%d")


def main(argv):
    runfile = ""
    qrelfile = ""
    collection = "robust"
    dump_rel = False
    pd = 100
    try:
        opts, args = getopt.getopt(argv, "r:c:j:d:bh",
                                   ["runlist", "collection",
                                    "qrelfile","depth"
                                    "backgain","help"])
    except getopt.GetoptError:
        print('-j <qrelfile> -c <collection> -j <qrelfile> '
              '-d <pooling depth> -b backgain -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-j <qrelfile> -c <collection> -j <qrelfile> '
              '-d <pooling depth> -b backgain -h help')
            sys.exit()
        elif opt in ("-j", "--jfile"):
            qrelfile = arg
        elif opt in ("-c", "--collection"):
            collection = arg
        elif opt in ("-r", "--run list"):
            runfile = arg
        elif opt in ("-b", "--backgain"):
            dump_rel = True
        elif opt in ("-d", "--depth"):
            pd = int(arg)

    # because I am lazy to rename.. not necessary
    cstr = "trec2006" if collection == "tb06" else collection
    if collection=="tb04":
        cstr="trec2004"
    elif collection=="tb05":
        cstr="trec2005"

    dirstr = "/research/remote/petabyte/users/xiaolul/syncbitbucket/" \
             "cluewebap/sigirexp/trec_order/original/" + cstr + "/"
    if collection=="robust" or collection == "tb06":
        dirstr = "/research/remote/petabyte/users/xiaolul/syncbitbucket/" \
                 "cluewebap/sigirexp/rank_order/original/" + cstr + "/"

    qrels = Qrel(qrelfile)
    runlist = []
    with open(runfile,'rb') as fin:
        for rname in fin:
            runlist.append(Qres(dirstr + rname.strip()))
    # start to dump rank
    dump_rank = DumpRank(runlist=runlist, qrels=qrels)
    if not dump_rel:
        out_pref = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
                   + collection + "/doc_rank/"
        dump_rank.dump_rank(out_pref)
    else:
        out_pref = "/research/remote/petabyte/users/xiaolul/pool_probability/" \
                   + collection + "/rank_rel/"
        dump_rank.dump_rel(out_pref,pd)

if __name__ == "__main__":
    main(sys.argv[1:])