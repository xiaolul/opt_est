import sys
import re
from collections import OrderedDict
from collections import defaultdict


class ResTpl:
    """class of qres tuple. it looks like
    51 Q0 clueweb09-en0007-17-32780 1 -8.3485 cmuFuTop10
    qid dummy docid rank score runid"""

    def __init__(self, str):
        """
        Initlize qres tuple
        :param str: line in the qres file
        :return: null
        """
        str_arr = str.strip().split()
        # print str_arr
        if len(str_arr) < 6:
            print (sys.stderr, 'Error in format')
        self._qid = int(str_arr[0])
        self._doc = str_arr[2]
        self._rank = str_arr[3]
        # MPI has 1{,}000 as a rank number
        self._rank = re.sub('[^0-9]', '', self._rank)
        self._score = str_arr[4]
        self._runid = str_arr[5]

    def set_score(self, score):
        """
        set score for current tuple
        :param score: score of current document
        :return:
        """
        self._score = score

    def set_rank(self, rank):
        """
        Set rank for current document
        :param rank:
        :return:
        """
        self._rank = rank

    def get_score(self):
        """
        get score of current document
        :return: score of the document
        """
        return float(self._score)

    def get_rank(self):
        """
        get rank of the document
        :return: rank of document
        """
        return int(self._rank)

    def get_qid(self):
        """
        get qid
        :return:
        """
        return int(self._qid)

    def get_doc(self):
        """
        get doc id in order to generate trec order
        :return:
        """
        return self._doc

    def get_str(self):
        tpl_str = str(self._qid) + " Q0 " + self._doc + " " + str(self._rank) + " " + str(
            self._score) + " " + self._runid
        return tpl_str.strip()

    def __hash__(self):
        return hash((str(self._qid) + self._doc).strip())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._qid == other.get_qid() and \
               self._doc == other.get_doc()


class Qres:
    """Class for qres submitted to TREC"""

    def __init__(self, fname):
        """
        initlize a qres from fname
        :rtype: object
        :param fname:
        :return:
        """
        self._rid = fname
        tmpdict = defaultdict(list)
        self._ties = {}
        with open(fname, "r") as fin:
            for line in fin:
                if len(line.strip()) > 1:
                    tpl = ResTpl(line)
                    curr_q = tpl.get_qid()
                    if (curr_q not in tmpdict) or len(tmpdict[curr_q]) < 1000:
                        # force to load only top 1000
                        tmpdict[curr_q].append(tpl)
        fin.close()
        self._run = OrderedDict(sorted(tmpdict.items(), reverse=False))

    def _check_ties(self, rank):
        """
        check ties up to a rank
        :param rank:up to which we will get
        :return:
        """
        for k in self._run:
            curr_dict = defaultdict(list)
            qid = self._run[k][0].get_qid()
            for t in self._run[k]:
                # print t.get_str()
                curr_dict[t.get_score()].append(t)
            curr_dict = OrderedDict(sorted(curr_dict.items(), reverse=True))
            cnt = 0
            for score in curr_dict:
                cnt = cnt + 1
                if len(curr_dict[score]) > 1 and cnt < rank:
                    if qid in self._ties.keys():
                        self._ties[qid] += len(curr_dict[score]) / float(rank)
                    else:
                        self._ties[qid] = len(curr_dict[score]) / float(rank)

    def get_ties(self, rank):
        """
        get tied numbers up to a rank
        :param rank:
        :return:
        """
        self._check_ties(rank)
        for k, v in self._ties.iteritems():
            print (str(k) + "," + str(v))

    def set_rank_order(self):
        """
        alter the scores based on rank
        and print out the qres string
        :return:
        """
        for k in self._run:
            # reset the score based on doc occurr. ignore rank
            tot_res = len(self._run[k])
            for r in self._run[k]:
                r.set_score(tot_res)
                tot_res -= 1
            self._run[k].sort(key=lambda x: x.get_score(), reverse=True)
            for r in self._run[k]:
                print (r.get_str())

    def set_trec_order(self):
        """
        set the qres order in trec way, tie breaking by inverse doc id.
        But we do not want to keep tied score there, so the score will be reassigned
        based on the rank
        :return:
        """
        for k in self._run:
            self._run[k].sort(key=lambda x: (x.get_score(), x.get_doc()), reverse=True)
            tot_rank = len(self._run[k])
            for r in self._run[k]:
                r.set_score(tot_rank)
                tot_rank -= 1
                print (r.get_str())

    def get_qid_set(self):
        """
        get all the qids
        :return: qids avaiable in current run
        """
        return self._run.keys()

    def get_res_by_id(self, qid):
        """
        the current res of the given qid
        :param qid:
        :return:
        """
        if qid not in self._run:
            return {}
        else:
            return self._run[qid]

    def get_res_set(self, qid, depth):
        sys_res = []
        limit_depth = len(self._run[qid]) if len(self._run[qid]) < depth else depth
        for i in range(0, limit_depth):
            sys_res.append(self._run[qid][i].get_doc())
        return sys_res

    def get_score_lst(self, qid):
        score_lst = []
        for i in range(0, len(self._run[qid])):
            score_lst.append(self._run[qid][i].get_score())
        return score_lst

    def delete_res_by_qid(self, qid):
        """
        delete qres by qid. Otherwise memory consumptions can be high
        :param qid:
        :return:
        """
        del self._run[qid]

if __name__ == "__main__":
    # use command line as input
    fname = sys.argv[1]
    tie_rank = sys.argv[2]
    opt = sys.argv[3]
    qres = Qres(fname)
    if opt == 'c':
        # check ties only
        qres.get_ties(int(tie_rank))
    elif opt == 't':
        qres.set_trec_order()
    elif opt == 'o':
        qres.set_rank_order()
