from qres_chk import Qres
from qrel_chk import Qrel
import numpy as np
import sys
import getopt
import futils

class ContribFinder:
    """
    Finding contributing runs based on input
    run list, qrels and pooling depth
    """
    def __init__(self, dirstr, runlist, qrelfname):
        """
        init the finder
        :param dirstr: dir where those runs located
        :param runlist: list of run NAMES ONLY
        :param qrelfname: qrel name
        :param pd: pooling depth
        """
        self._dir = dirstr
        self._qrel = Qrel(qrelfname)
        self._qid = self._qrel.get_qid()
        self._runlist = runlist

    def find_contrib(self, depth,name):
        """
        find contributing runs based on input pooling depth
        :param depth:
        :return:
        """
        fout = open( name + ".contribute.run","a")
        with open(self._runlist, 'rb') as fin:
            for lines in fin:
                print (lines)
                is_contrib = True
                qres = Qres(self._dir + lines.strip())
                for q in self._qid:
                    curr_j = self._qrel.get_judged_by_qid(q)
                    curr_res = qres.get_res_by_id(q)
                    if len(curr_res) == 0:
                        print ("No query " + str(q))
                        is_contrib = False
                    last_pos = depth if len(curr_res) > depth else len(curr_res)
                    for i in range(0, last_pos):
                        curr_doc = curr_res[i].get_doc()
                        if curr_doc not in curr_j:
                            print (str(i))
                            is_contrib = False
                            break
                    if not is_contrib:
                        break
                if is_contrib:
                    fout.write(lines)

    def dump_rank(self,in_dir, out_dir, depth, collection):
        """
        dump out the minimal ranking for documents in the qrel file
        :param in_dir run_file.
        :param out_dir out_dir
        :param depth. pooling depth
        :param collection document collection
        :return:
        """
        #
        contrib_list = []
        fout = open(out_dir + collection + ".contribute.run", "a")
        with open(self._runlist, 'rb') as fin:
            for lines in fin:
                print (lines)
                is_contrib = True
                qres = Qres(self._dir + lines.strip())
                for q in self._qid:
                    curr_j = self._qrel.get_judged_by_qid(q)
                    curr_res = qres.get_res_by_id(q)
                    last_pos = depth if len(curr_res) > depth else len(curr_res)
                    for i in range(0, last_pos):
                        curr_doc = curr_res[i].get_doc()
                        if curr_doc not in curr_j:
                            print (str(i))
                            is_contrib = False
                            break
                    if not is_contrib:
                        break
                if is_contrib:
                    #assert code here if you want to dump file only
                    print (lines.strip())
                    fout.write(lines)
                    contrib_list.append(lines.strip())
        fout.close()
        rank_out = open(out_dir + collection + ".rank.qrel", "a")
        for q in self._qid:
            curr_dict = {}
            for r in contrib_list:
                qres = Qres(self._dir + r)
                curr_j = self._qrel.get_rel_by_qid(q)
                curr_res = qres.get_res_by_id(q)
                for i in range(0, len(curr_res)):
                    curr_doc = curr_res[i].get_doc()
                    if curr_doc in curr_j:
                        if curr_doc in curr_dict:
                            curr_dict[curr_doc] = min(curr_dict[curr_doc], (i+1))
                        else:
                            curr_dict[curr_doc] = (i+1)
            for k, v in curr_dict.iteritems():
                rank_out.write(str(q) + ", " + str(k) + ", " + str(v) + "\n")
        rank_out.close()

    def trim_qrel(self, d, out_dir):
        run_list = []
        with open(self._runlist, "r") as fin:
            for line in fin:
                run_list.append(line.strip())
        fout = open(out_dir + "qrels."+str(d)+".txt","a")
        for q in self._qid:
            curr_qrel = self._qrel.get_rel_by_qid(q)
            curr_dic = {}
            for r in run_list:
                qres = Qres(self._dir + r)
                curr_res = qres.get_res_by_id(q)
                last_pos = d if len(curr_res) > d else len(curr_res)
                for i in range(0, last_pos):
                    curr_doc = curr_res[i].get_doc()
                    if curr_doc in curr_qrel:
                        if curr_doc not in curr_dic:
                            fout.write(str(q) + " 0 " + curr_doc + " " + str(curr_qrel[curr_doc]) + "\n")
        fout.close()


def dump_gidx(run_list, output_dir, gname):
    """
    dump the idx of each run
    :param run_list: list of runs names [str]
    :param output_dir: dir of the run list. trec order
    :param gname: list of group name
    :return: dict{} of run and idx. dumping it as well
    """
    gidx = {}
    with open(gname, "rb") as fin:
        for line in fin:
            gidx[line.strip()] = []
            for i in range(0, len(run_list)):
                rname = run_list[i].split(".")[1]
                if rname.startswith(line.strip()):
                    print (run_list[i], line.strip())
                    gidx[line.strip()].append(i)
    for k, v in gidx.iteritems():
        with open(output_dir+"input."+ k + ".txt", "a") as fout:
            for idx in v:
                fout.write(str(idx)+"\n")
    return gidx


def dump_log_qrel(qrelfile, output_dir, gidx, rank_dir, dmax):
    """
    dump step wise log qrels
    :param qrelfile: *Full* qrels
    :param output_dir: output dir
    :param gidx: group idx
    :param rank_dir: rank dir
    :param dmax: maximum considered pooling depth
    :return:
    """
    dmax += 10
    qrel = Qrel(qrelfile)
    qid = qrel.get_qid()
    dlist = range(10, dmax, 10)
    for k, v in gidx.iteritems():
        log_idx = np.array(v)
        for i in range(0, len(qid)):
            curr_qrel = qrel.get_rel_by_qid(qid[i])
            rank_mat, runum = futils.read_csv_to_dict(rank_dir+str(qid[i])+"-rank.txt", is_prob=False)
            for doc, rank in rank_mat.iteritems():
                tmp_rnk = np.array(rank)
                if doc in curr_qrel:
                    rel_val = curr_qrel[doc]
                    log_rnk = np.delete(tmp_rnk, log_idx)
                    log_rnk = log_rnk[[log_rnk > -1]]  # only care about this
                    if len(log_rnk) > 0:
                        if min(log_rnk) < dlist[-1]:
                            for d in dlist:
                                if min(log_rnk) < d:
                                    fname = output_dir+str(d)+"/qrels.input."+k+".txt"
                                    with open(fname, "a") as fout:
                                        fout.write(str(qid[i]) + " 0 " + doc + " " + str(rel_val) + "\n")

def main(argv):
    runfile = ""
    qrelfile = ""
    outputfile = ""
    depth = 40
    collection = "robust"
    is_dump_gidx = False
    pd = 100
    try:
        opts, args = getopt.getopt(argv, "hr:j:d:c:g", ["runf", "jfile", "depth"])
    except getopt.GetoptError:
        print('inf_pool.py -r <runlist> -j <qrelfile> -c <collection> -d <depth> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('inf_pool.py -r <runlist> -j <qrelfile> -o <output> -d <depth> -g -h help')
            sys.exit()
        elif opt in ("-r", "--runf"):
            runfile = arg
        elif opt in ("-j", "--jfile"):
            qrelfile = arg
        elif opt in ("-c", "--ofile"):
            collection = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-g"):
            is_dump_gidx = True

    cstr = collection
    if collection == "tb04":
        cstr = "trec2004"
        pd = 80
    elif collection == "tb05":
        cstr = "trec2005"
    dirstr = "/research/remote/petabyte/users/xiaolul/syncbitbucket/cluewebap/sigirexp/trec_order/original/" + cstr + "/"
    if collection == "tb06":
        dirstr = "/research/remote/petabyte/users/xiaolul/syncbitbucket/cluewebap/sigirexp/rank_order/original/" + collection + "/"
        pd = 50

    out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/" + collection + "/qrels/"
    if is_dump_gidx:
        runlist = []
        out_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+collection+"/qrels/log_qrels/"
        gname = "/research/remote/petabyte/users/xiaolul/pool_probability/contrib_run/"+collection+".gname"
        rank_dir = "/research/remote/petabyte/users/xiaolul/pool_probability/"+collection+"/doc_rank/"
        with open(runfile, "rb") as fin:
            for line in fin:
                runlist.append(line.strip())
        gidx = dump_gidx(run_list=runlist, output_dir=out_dir, gname=gname)
        dump_log_qrel(qrelfile=qrelfile, output_dir=out_dir, gidx=gidx, rank_dir=rank_dir, dmax=pd)
    else:
        finder = ContribFinder(dirstr,runfile,  qrelfile)
        finder.trim_qrel(d=depth, out_dir=out_dir)
        # finder.find_contrib(depth, collection)
        # finder.dump_rank(dirstr,out_dir,depth,collection)


if __name__ == "__main__":
    main(sys.argv[1:])