from __future__ import division
import numpy as np
from scipy import stats
import sys
import getopt

def tau_dist(fname, refmat, depth, out_dir, qnum):
    """
    Normalized tau distance, counting inversed pairs only.
    :param curr_mat:
    :param ref_mat:
    :param depth:
    :param out_dir:
    :param qnum:
    :return:
    """
    ref_point = np.array(refmat)  # reference point
    sys_num = ref_point.shape[1]  # number of systems
    curr_mat = np.loadtxt(fname, delimiter=",", dtype=float)  # load current score
    mnum = curr_mat.shape[1]
    res_mat = np.zeros((mnum, 3))  # result matrix
    res_mat[:, 0] = range(0, mnum)
    res_mat[:, -1] = depth
    for i in range(0, mnum):
        curr_method = curr_mat[:, i]  # reshape vector to q*n matrix
        curr_score = np.zeros((qnum, sys_num))
        t = 0
        while t < sys_num:
            curr_score[:, t] = curr_method[t * qnum:(t + 1) * qnum]
            t += 1
        for n in range(0, (sys_num - 1)):
            for s in range(n + 1, sys_num):
                t_stat, p_val = stats.ttest_rel(curr_score[:, n],
                                                curr_score[:, s])
                if (np.sign(t_stat) < 0 and ref_point[n, s] != 0) \
                        or (np.sign(t_stat) > 0 and ref_point[s, n] != 0):
                    res_mat[i, 1] += 1
    res_mat[:,1] /= (sys_num * (sys_num - 1) * 0.5)
    out_name = out_dir + "dist-" + str(depth) + "-tau.txt"
    np.savetxt(out_name, X=res_mat, fmt="%d %.4f %d",
               delimiter=",",
               header="method dist depth")



def sig_dist(fname, refmat, depth, out_dir, qnum):
    """
    calculte distance based on significance testing
    :param fname: file name of the current score matrix
    :param refmat: reference matrix
    :param depth: depth, for naming the result file
    :param out_dir: output dir
    :param qnum: query numbers.
    :return:
    """
    ref_point = np.array(refmat)  # reference point
    sys_num = ref_point.shape[1]  # number of systems
    curr_mat = np.loadtxt(fname, delimiter=",", dtype=float)  # load current score
    mnum = curr_mat.shape[1]
    res_mat = np.zeros((mnum, 3))  # result matrix
    res_mat[:, 0] = range(0, mnum)
    res_mat[:, -1] = depth
    for i in range(0, mnum):
        curr_method = curr_mat[:, i]  # reshape vector to q*n matrix
        curr_score = np.zeros((qnum, sys_num))
        qlist = range(0, len(curr_method), qnum)
        for q in range(0, len(qlist)):
            curr_score[:, q] = curr_method[qlist[q]:(qlist[q] + qnum)]
        for n in range(0, (sys_num - 1)):
            for s in range(n + 1, sys_num):
                t_stat, p_val = stats.ttest_rel(curr_score[:, n],
                                                curr_score[:, s])
                m_ij = 0
                ref_val = 0
                if np.mean(curr_score[:, n]- curr_score[:, s]) != 0:
                    if t_stat > 0:
                        m_ij = (1 - p_val) * 0.5
                    else:
                        m_ij = (p_val - 1 )* 0.5
                if ref_point[n, s] != 0:
                    ref_val = (1 - p_val) * 0.5
                elif ref_point[s, n] != 0:
                    ref_val = (p_val - 1 ) * 0.5
                res_mat[i, 1] += abs(m_ij - ref_val)
    out_name = out_dir + "dist-" + str(depth) + ".txt"
    np.savetxt(out_name, X=res_mat, fmt="%d %.4f %d",
               delimiter=",",
               header="method dist depth")

def sample_sim_calc(ref_dir, input_dir,collection, rnd, out_dir):
    """
    calculate distance for the sampling based methods.
    :param ref_dir: reference dir trec/rbp
    :param input_dir: inf_hybrid or sample_eval
    :param collection: collection name. robust, tb04, tb05 etc.
    :param rnd: rnd number
    :return:
    """
    ref_score = np.loadtxt(ref_dir+ "lb.txt",dtype=float)
    lb_score = np.loadtxt(ref_dir+ "lb."+str(rnd)+".txt",dtype=float)
    rm_score = np.loadtxt(ref_dir + "rm." + str(rnd) + ".txt", dtype=float)
    opta_score = np.loadtxt(input_dir + "hybrid/summary/opt."+str(rnd)+".txt",dtype=float)
    optb_score = np.loadtxt(input_dir + "hybrid/summary/optd."+str(rnd)+".txt",dtype=float)
    inf_score = np.loadtxt(input_dir + "sample_eval/summary/inf."+str(rnd)+".txt",dtype=float)
    sys_num = ref_score.shape[1]
    # lb, rm, inf, opta, optb
    tau_dist = np.zeros(5)
    w_dist = np.zeros(5)
    tau_cor = np.zeros(5)
    for i in range(0, (sys_num-1)):
        i_ref = ref_score[:, i]
        i_lb = lb_score[:, i]
        i_rm = rm_score[:, i]
        i_opta = opta_score[:, i]
        i_optb = optb_score[:, i]
        i_inf = inf_score[:, i]
        for j in range(i+1, sys_num):
            j_ref = ref_score[:, j]
            j_lb = lb_score[:, j]
            j_rm = rm_score[:, j]
            j_opta = opta_score[:, j]
            j_optb = optb_score[:, j]
            j_inf = inf_score[:, j]
            ref_sign, ref_val =get_sign_pval(i_ref, j_ref)
            lb_sign, lb_val = get_sign_pval(i_lb, j_lb)
            rm_sign, rm_val = get_sign_pval(i_rm, j_rm)
            opta_sign, opta_val = get_sign_pval(i_opta, j_opta)
            optb_sign, optb_val = get_sign_pval(i_optb, j_optb)
            inf_sign, inf_val = get_sign_pval(i_inf, j_inf)
            if np.sign(ref_sign * lb_sign) < 0:
                tau_dist[0] += 1
            if np.sign(ref_sign * rm_sign) < 0:
                tau_dist[1] += 1
            if np.sign(ref_sign * inf_sign) < 0:
                tau_dist[2] += 1
            if np.sign(ref_sign * opta_sign) < 0:
                tau_dist[3] += 1
            if np.sign(ref_sign * optb_sign) < 0:
                tau_dist[4] += 1
            w_dist[0] += np.abs(lb_val - ref_val)
            w_dist[1] += np.abs(rm_val - ref_val)
            w_dist[2] += np.abs(inf_val - ref_val)
            w_dist[3] += np.abs(opta_val - ref_val)
            w_dist[4] += np.abs(optb_val - ref_val)
    ref_avg = np.mean(ref_score, axis= 0)
    tau_cor[0], pval = stats.kendalltau( np.mean(lb_score, axis= 0), ref_avg)
    tau_cor[1], pval = stats.kendalltau(np.mean(rm_score, axis=0), ref_avg)
    tau_cor[2], pval = stats.kendalltau(np.mean(inf_score, axis=0), ref_avg)
    tau_cor[3], pval = stats.kendalltau(np.mean(opta_score, axis=0), ref_avg)
    tau_cor[4], pval = stats.kendalltau(np.mean(optb_score, axis=0), ref_avg)
    tau_dist /= (sys_num * (sys_num - 1)) * 0.5
    with open(out_dir + "rnd-dist.txt", "a") as fout:
        fout.write("#m,tau,dist,wdist,rnd,collection\n")
        for i in range(0, 5):
            fout.write(str(i) + ",{:.4f},{:.4f},{:.4f},{:d},".format(tau_cor[i],tau_dist[i],w_dist[i],int(rnd))+collection+"\n")

def inf_sim_calc(ref_dir,collection, rnd, input_dir=""):
    """
    calculate distance for the sampling based methods.
    :param ref_dir: reference dir trec/rbp
    :param input_dir: inf_hybrid or sample_eval
    :param collection: collection name. robust, tb04, tb05 etc.
    :param rnd: rnd number
    :return:
    """
    if input_dir == "":
        input_dir = ref_dir
    ref_score = np.loadtxt(ref_dir+ "lb.txt",dtype=float)
    lb_score = np.loadtxt(input_dir+ "inf."+str(rnd)+".txt",dtype=float)
    sys_num = ref_score.shape[1]
    # lb, rm, inf, opta, optb
    tau_dist = 0
    w_dist = 0
    tau_cor = 0
    for i in range(0, (sys_num-1)):
        i_ref = ref_score[:, i]
        i_lb = lb_score[:, i]
        for j in range(i+1, sys_num):
            j_ref = ref_score[:, j]
            j_lb = lb_score[:, j]
            ref_sign, ref_val =get_sign_pval(i_ref, j_ref)
            lb_sign, lb_val = get_sign_pval(i_lb, j_lb)
            if np.sign(ref_sign * lb_sign) < 0:
                tau_dist += 1
            w_dist += np.abs(lb_val - ref_val)
    ref_avg = np.mean(ref_score, axis= 0)
    tau_cor, pval = stats.kendalltau( np.mean(lb_score, axis= 0), ref_avg)
    tau_dist /= (sys_num * (sys_num - 1)) * 0.5
    with open(input_dir + "rnd-dist.txt", "a") as fout:
        fout.write("2,{:.4f},{:.4f},{:.4f},{:d},".format(tau_cor, tau_dist,
                                                          w_dist, int(rnd)) + collection + "\n")

def inf_discr_calc(rnd, input_dir):
    lb_score = np.loadtxt(input_dir + "optd." + str(rnd) + ".txt", dtype=float)
    sys_num =lb_score.shape[1]
    pval_list = []
    for i in range(0, (sys_num-1)):
        sys_i = lb_score[:,i]
        for j in range(i+1, sys_num):
            sys_j = lb_score[:, j]
            t_stat, pval = stats.ttest_rel(sys_i, sys_j)
            if not np.isnan(t_stat):
                pval_list.append(pval)
    print np.median(np.array(pval_list))


def cov_inv_calc(ref_dir, rnd, input_dir=""):
    """
    calculate coverage ratio
    :param ref_dir: reference dir
    :param rnd: random number
    :param input_dir:
    :return:
    """
    ref_score = np.loadtxt(ref_dir+ "lb.txt",dtype=float)
    sys_num = ref_score.shape[1]
    tot_cnt = sys_num * (sys_num - 1) * 0.5
    lb_score = np.loadtxt(input_dir + "optd." + str(rnd) + ".txt", dtype=float)
    cov_cnt = 0
    inv_cnt = 0
    for i in range(0, (sys_num-1)):
        ref_i = ref_score[:, i]
        curr_i  = lb_score[:, i]
        for j in range(i+1, sys_num):
            ref_j = ref_score[:, j]
            curr_j = lb_score[:, j]
            ref_stat, ref_pval  = stats.ttest_rel(ref_i, ref_j)
            curr_stat, curr_pval = stats.ttest_rel(curr_i, curr_j)
            if np.sign(curr_stat * ref_stat) > 0:
                if curr_pval < ref_pval <= 0.05:
                    cov_cnt += 1
            elif np.sign(curr_stat * ref_stat) < 0:
                if ref_pval <= 0.05:
                    inv_cnt += 1
    print "{:.2f},{:.2f}".format(100*cov_cnt/(tot_cnt), 100*inv_cnt/(tot_cnt))




def get_sign_pval(arr_a, arr_b):
    """
    two tail paired t-test, return weight and sign
    :param arr_a:
    :param arr_b:
    :return:
    """
    ret_sign =0
    ret_p = 0
    if np.mean(arr_a) - np.mean(arr_b) != 0:
        t_stat, pval = stats.ttest_rel(arr_a, arr_b)
        if t_stat > 0:
            ret_p = (1 - pval) * 0.5
            ret_sign = 1
        else:
            ret_p = (pval - 1) * 0.5
            ret_sign = -1
    return ret_sign, ret_p


def main(argv):
    # ---default values
    collection = "trec10"
    dir_str = collection + "/est_eval/fit_eval/origin/summary/combined."
    tmp_out = ""
    # depth = 50
    suf = "rbp"
    pd = 100
    try:
        opts, args = getopt.getopt(argv, "c:p:d:s:q:h", ["collection", "path", "depth", "suf", "help"])
    except getopt.GetoptError:
        print('-c <collection> -p <path> -d <depth>  -s <suffix> -h help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-j <qrelfile> -d <depth> -c <collection> -h help')
            sys.exit()
        elif opt in ("-c", "--collection"):
            collection = arg
        elif opt in ("-p", "--path"):
            dir_str = arg
        elif opt in ("-d", "--file"):  # original pooling depth
            pd = arg
        elif opt in ("-s", "--file"):
            suf = arg


    # #------calculate distance for the pooling method
    suf = "." + suf
    t = 50  # number of topics
    # pref_str = dir_str
    refname = dir_str + str(pd) + suf
    tmp_ref = np.loadtxt(refname, dtype=float, delimiter=",")  # first column is lb
    # mnum = tmp_ref.shape[1]
    sys_num = int(tmp_ref.shape[0] / t)
    tmp_ref = tmp_ref[:, 0]
    #
    ref_mat = np.zeros((t, sys_num))
    i = 0
    while i < sys_num:
        ref_mat[:, i] = tmp_ref[i * t:(i + 1) * t]
        i += 1
    ref_pmat = np.zeros((sys_num, sys_num))
    for i in range(0, (sys_num - 1)):
        for j in range((i + 1), sys_num):
            t_stat, pval = stats.ttest_rel(ref_mat[:, i],
                                           ref_mat[:, j])
            if np.mean(ref_mat[:, i] - ref_mat[:, j]) != 0:
                if t_stat > 0:
                    ref_pmat[i, j] = pval
                else:
                    ref_pmat[j, i] = pval
    # compute distance for rank-level estimators, without threshold
    for k in range(10, pd + 1, 10):
        sig_dist(dir_str + str(k) + suf, ref_pmat, k, tmp_out, t)
        tau_dist(dir_str + str(k) + suf, ref_pmat, k, tmp_out, t)


if __name__ == "__main__":
    main(sys.argv[1:])
