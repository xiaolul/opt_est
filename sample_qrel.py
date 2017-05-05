from qres_chk import Qres
from qrel_chk import Qrel
import numpy as np
import sys, getopt

class InfAP:
    """
    simulate the sampling process from paper
    sigir-08:
    A simple and efficient sampling method for
    estimating AP and NDCG
    the method exhaustively pool to some shallow
    depth and then perform random sample the same
    amount of documents
    Notice that a binary relevance is used.
    RELVALUE UNJUDGED (-1)  if doc no in text qrels, but not judged, marked as -1.
    This is so called ``pooled but unjudged"
    """
    def __init__(self, qrel_name, run_list):
        self._runlist = []
        for f in run_list:
            self._runlist.append(Qres(f))
        self._qrel = Qrel(qrel_name)
        self._qid = self._qrel.get_qid()
        self._run_num = len(self._runlist)

    def sample_doc(self, init_rank, out_dir, max_d = 100):
        """
        simulate the experiments done by E. Yamlaz
        :param init_rank:
        :param out_dir:
        :param max_d:
        :return:
        """
        for q in self._qid:
            judged_cnt = 0
            rel_cnt = 0
            qrel_dict = {}
            curr_rel =self._qrel.get_rel_by_qid(q)
            doc_set = set()
            for i in range(0, self._run_num):
                curr_res = self._runlist[i].get_res_by_id(q)
                init_pos = init_rank if len(curr_res) > init_rank else len(curr_res)
                for d in range(0, init_pos):
                    curr_doc = curr_res[d].get_doc()
                    if curr_doc not in qrel_dict:
                        if curr_doc in doc_set:  # doc_set contains prev docs have larger rank.
                            doc_set.remove(curr_doc)  # need to remove
                        qrel_dict[curr_doc] = 0
                        if curr_doc in curr_rel and \
                                        curr_rel[curr_doc] > 0:
                                qrel_dict[curr_doc] = 1
                                rel_cnt += 1
                last_rank = max_d if len(curr_res) > max_d else len(curr_res)
                for d in range(init_pos, last_rank):
                    candidate_doc = curr_res[d].get_doc()
                    if candidate_doc not in qrel_dict:
                        doc_set.add(candidate_doc)
            judged_cnt = len(qrel_dict)
            doc_set = list(set(doc_set))  # de-duplicate
            sample_size = len(qrel_dict) if len(qrel_dict) < len(doc_set) else len(doc_set)
            # sample same amount of documents as in the top-k pool
            sample_idx = range(0, len(doc_set))
            if sample_size < len(doc_set):
                sample_idx = np.random.choice(np.array(range(0, len(doc_set))),
                                              size=sample_size, replace=False)
            sample_idx = set(sample_idx)
            for doc_idx in range(0, len(doc_set)):
                if doc_idx in sample_idx:
                    judged_cnt += 1
                    qrel_dict[doc_set[doc_idx]] = 0
                    if doc_set[doc_idx] in curr_rel and \
                         curr_rel[doc_set[doc_idx]] > 0:
                            qrel_dict[doc_set[doc_idx]] = 1
                            rel_cnt += 1
                    else:
                        qrel_dict[doc_set[doc_idx]] = 0
                else:
                    qrel_dict[doc_set[doc_idx]] = - 1   # pooled but unjudged
            with open(out_dir+"qrel."+str(init_rank)+".txt", "a") as fout:
                for doc, rel in qrel_dict.iteritems():
                    fout.write(str(q) + " 0 " + doc + " " + str(rel) + "\n")
            with open(out_dir+"qrel."+str(init_rank)+"-ts.txt", "a") as fout:
                for doc, rel in qrel_dict.iteritems():
                    if rel != -1:
                        fout.write(str(q) + " 0 " + doc + " " + str(rel) + "\n")
            with open(out_dir+"stat."+str(init_rank)+".txt", "a") as fout:
                fout.write(str(q) + "," + str(judged_cnt) + "," + str(rel_cnt) + "\n")

    def two_strata_sample(self,out_dir, epoch_num, sample_rate = 0.1, max_d = 100):
        """
        sample documents using Voorhees method
        :param out_dir:
        :param epoch_num: which round the sample is at
        :param sample_rate: sample rate, 0.1 by default.
        :param max_d: maximum pooling depth
        :return:
        """
        init_rank = 10
        for q in self._qid:
            judged_cnt = 0
            rel_cnt = 0
            qrel_dict = {}
            curr_rel =self._qrel.get_rel_by_qid(q)
            doc_set = set()
            for i in range(0, self._run_num):
                curr_res = self._runlist[i].get_res_by_id(q)
                init_pos = init_rank if len(curr_res) > init_rank else len(curr_res)
                for d in range(0, init_pos):
                    curr_doc = curr_res[d].get_doc()
                    if curr_doc not in qrel_dict:
                        if curr_doc in doc_set:  # doc_set contains prev docs have larger rank.
                            doc_set.remove(curr_doc)  # need to remove
                        qrel_dict[curr_doc] = 0
                        if curr_doc in curr_rel and \
                                        curr_rel[curr_doc] > 0:
                                qrel_dict[curr_doc] = 1
                                rel_cnt += 1
                last_rank = max_d if len(curr_res) > max_d else len(curr_res)
                for d in range(init_pos, last_rank):
                    candidate_doc = curr_res[d].get_doc()
                    if candidate_doc not in qrel_dict:
                        doc_set.add(candidate_doc)
            judged_cnt = len(qrel_dict)
            doc_set = list(set(doc_set))  # de-duplicate
            sample_size = sample_rate * len(doc_set)
            if sample_size > len(doc_set):
                sample_size = len(doc_set)
            sample_size = int(np.floor(sample_size))
            # sample same amount of documents as in the top-k pool
            sample_idx = range(0, len(doc_set))
            if sample_size < len(doc_set):
                sample_idx = np.random.choice(np.array(range(0, len(doc_set))),
                                              size=sample_size, replace=False)
            sample_idx = set(sample_idx)
            for doc_idx in range(0, len(doc_set)):
                if doc_idx in sample_idx:
                    judged_cnt += 1
                    qrel_dict[doc_set[doc_idx]] = 0
                    if doc_set[doc_idx] in curr_rel and \
                         curr_rel[doc_set[doc_idx]] > 0:
                            qrel_dict[doc_set[doc_idx]] = 1
                            rel_cnt += 1
                    else:
                        qrel_dict[doc_set[doc_idx]] = 0
                else:
                    qrel_dict[doc_set[doc_idx]] = - 1   # pooled but unjudged
            with open(out_dir+"qrel."+str(epoch_num)+".txt", "a") as fout:
                for doc, rel in qrel_dict.iteritems():
                    fout.write(str(q) + " 0 " + doc + " " + str(rel) + "\n")
            with open(out_dir+"qrel."+str(epoch_num)+"-ts.txt", "a") as fout:
                for doc, rel in qrel_dict.iteritems():
                    if rel != -1:
                        fout.write(str(q) + " 0 " + doc + " " + str(rel) + "\n")
            with open(out_dir+"stat."+str(epoch_num)+".txt", "a") as fout:
                fout.write(str(q) + "," + str(judged_cnt) + "," + str(rel_cnt) + "\n")


def main(argv):
    qres_dir = ""
    qrelfile = ""
    depth = 10
    collection = "robust"
    runfile = ""
    pd = 100

    try:
        opts, args = getopt.getopt(argv, "r:j:c:d:", ["runf", "jfile", "collection","init_d"])
    except getopt.GetoptError:
        print('-r <runlist> -j <qrelfile> -d <depth> -c <collection> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-r <runlist> -j <qrelfile> -d <depth> -c <collection> -h help')
            sys.exit()
        elif opt in ("-r", "--runf"):
            runfile = arg
        elif opt in ("-j", "--jfile"):
            qrelfile = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-c", "--depth"):
            collection = arg

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
        qres_dir = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                   "/cluewebap/sigirexp/rank_order/original/"+cstr + "/"
    else:
        qres_dir = "/research/remote/petabyte/users/xiaolul/syncbitbucket" \
                  "/cluewebap/sigirexp/trec_order/original/" + cstr + "/"

    qrel_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
               collection+"/qrels/"
    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+\
              collection+"/qrels/sample_qrels/"
    run_list = []
    with open(runfile, "rb") as fin:
        for lines in fin:
            run_list.append(qres_dir + lines.strip())
    inf_ap = InfAP(qrel_name=qrel_dir+qrelfile, run_list = run_list)
    for d in range(10,100):
        print (collection + " init "+ str(d))
        inf_ap.two_strata_sample(out_dir=out_dir,epoch_num=d, max_d=pd)

if __name__ == "__main__":
        main(sys.argv[1:])
