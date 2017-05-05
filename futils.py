import csv
from collections import defaultdict


def read_csv_to_dict(fname, is_prob=True):
    """
    read the document-rank matrix in to dictionary-list
    :param fname: path of the doc-rank file
    :param is_prob: if it is probability, map to float, else, map to int
    :return: doc_mat: default dict with document as key the rest as list,
            m: colnum
    """
    doc_mat = defaultdict(list)
    m = 0
    with open(fname, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")
        for row in reader:
            if m == 0:
                m = len(row) - 1
            if not is_prob:  # map to int
                doc_mat[row[0]] = [int(x) for x in row[1:]]
            else: # map to float.
                doc_mat[row[0]] = [float(x) for x in row[1:]]
    return doc_mat, m