from qrel_chk import Qrel
import sys, getopt

def train_gen(fname):
    """
    clean up the infap qrel...
    remove the -1 documents in case
    :param fname:
    :param out_dir:
    :return:
    """
    qrel_str = []
    with open(fname, "rb") as fin:
        for line in fin:
            curr_str = line.split()
            if int(curr_str[-1]) != -1:
                qrel_str.append(line.strip())
    with open(fname+"-ts", "w") as fout:
        for line in qrel_str:
            fout.write(line + "\n")

def sep_stratum(first_stratum, fname):
    """
    seperate the two-strata sampled qrels
    :param fname:
    :return:
    """
    qrel_str = []
    first_qrel = Qrel(first_stratum)
    sampled_qrel = Qrel(fname)
    qid = first_qrel.get_qid()
    for i in range(0, len(qid)):
        curr_qrel = first_qrel.get_rel_by_qid(qid[i])
        curr_sampled_qrel = sampled_qrel.get_rel_by_qid(qid[i])
        for doc, rel in curr_sampled_qrel.iteritems():
            if doc not in curr_qrel:
                qrel_str.append(str(qid[i]) + " 0 " + doc + " " + str(curr_sampled_qrel[doc]))
    with open(fname+"-2s", "w") as fout:
        for line in qrel_str:
            fout.write(line+ "\n")


def main(argv):
    fname = ""
    first_str  = ""
    try:
        opts, args = getopt.getopt(argv, "s:f:h",
                                   ["fname", "help"])
    except getopt.GetoptError:
        print('-f <fname>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-f <fname>')
            sys.exit()
        elif opt in ("-f", "--fname"):
            fname = arg
        elif opt in ("-s", "--first_stm"):
            first_str = arg

    # train_gen(fname=fname)
    sep_stratum(first_stratum=first_str, fname=fname)

if __name__ == "__main__":
    main(sys.argv[1:])