*The scripts haven't been cleaned yet.*

## Preliminaries
- All the estimations are based on contributing runs only.

- TREC relevance judgmeents are used.

- By default (Currently), they are in binary mode.

## Operations done by those scripts
1. Relevance fitting (fit_prob.py)
2. The two-level estimation framework (\*opt.py and hybrid_opt.py related scripts.)
  
## Scripts and Explaination (that have been cleaned now)
  1. fit_prob.py: model relevance as a function of retrieval rank.
     - By default, it is based on method in [1]
     - Left out run names can be provided and perform leave out experiments.
     - Open randomization can do a shuffle-and-average operation for the fitting process.
     
## Referece
[1] Lu, Xiaolu, Alistair Moffat, and J. Shane Culpepper. "Modeling relevance as a function of retrieval rank." Information Retrieval Technology. Springer International Publishing, 2016. 3-15.
