*The scripts haven't been (completely) cleaned yet.*

## Preliminaries
- All the estimations are based on contributing runs only.
- TREC relevance judgmeents are used.
- By default (Currently), they are in binary mode.
- scipy, numpy are required 

## Operations done by those scripts
1. Relevance fitting (fit_prob.py)
2. The two-level estimation framework (\*opt.py and hybrid_opt.py related scripts.)
3. There are two different configurations:
      1. Loss function:
        - Loss a: minimizing per-system estimation error (loss-a)
        - Loss b: minimizing per-document estimation error (loss-b)
      2. Combine function:
        - Weighted average (Comb1)
        - Weighted geometric average (Comb2)
  
## Scripts and Explaination (that have been cleaned now)
  1. dump_rank: rank-level back-ground estimations based on the input system matrix
  2. fit_prob.py: model relevance as a function of retrieval rank.
     - By default, it is based on method in [1]
     - Left out run names can be provided and perform leave out experiments.
     - Open randomization can do a shuffle-and-average operation for the fitting process.
     

  3. The first step optimizers in [2]
      - rbp_opt.py: loss-a and Comb1 
      - rbp_optd.py: loss-b and Comb1
      - rbp_opt_geo.py: loss-a and Comb2
      - rbp_optd_geo.py: loss-b and Comb2
      
  4. hybrid_opt_rbp.py: the second level rank estimator in [2] 
   
  5. naive_estimator.py: using coefficient of covariance in [3]
  
  6. In eval_tools directory:
     - goodness_of_fit.py: purely used for testing the rmse of curve fitting.
       The ref_dir should point to the relevance matrix where each row is the 
       relevance values returned by contributing systems and each column is a 
       ranked list.
     
     - kendall_dist.py: modified Kendall's distance defined in [2]
       Input estimated scores, reference score, original pooling depth and output dir.
       The estimated scores and reference score file should be formatted in the same way, 
       which is a sys_num * topic_num length vector.
       Topic numbers should also be specified.
   
 
  
   
  
  
## Input
 1. A set of contributing runs.
 2. The qrel file (which has the correct depth as specified)
 3. The rank-level estimations *must* run first as an input to the "*opt" estimators.
 4. In order to calculate the distance, all systems are required to be evaluated, and formated into a matrix, which has dimensions topic \times sys_num.

## Referece
[1] Xiaolu Lu, Alistair Moffat, and J. Shane Culpepper. "Modeling relevance as a function of retrieval rank." Information Retrieval Technology. Springer International Publishing, 2016. 3-15.

[2] Xiaolu Lu, Alistair Moffat, and J. Shane Culpepper. "Can deep metrics be evaluated using shallow judgment pools?" In Proc. SIGIR, 2017.

[3] A. Chao and S. Lee. Estimating the Number of Classes via Sample Coverage. Journal of the American Statistical Association, 87(417):210–217, 1992.