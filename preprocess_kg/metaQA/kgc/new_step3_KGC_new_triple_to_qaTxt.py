import os
import pickle
import argparse
# check the accuracy
import json


# this file is used to transform the triple to qa
# change 
"""
if it is 2hop, use:
    kgc_file = "./after_complete_2hop_kb.txt"
    file_name = "./after_complete_2hop_kb_qa.txt"

if it is 3hop, use:
    kgc_file = "./after_complete_3hop_kb.txt"
    file_name = "./after_complete_3hop_kb_qa.txt"


This file can also be used to transform the test.txt of KG
kgc_file = "./data/MetaQA_half/test.txt"
file_name = "./data/MetaQA_half/test_qa.txt"
"""

# this file only process metaQA 2hop and 3hop

def merge_kgc_with_QA_old(split=False):
    """
    if it is 2hop, use:
        kgc_file = "./after_complete_2hop_kb.txt"
        file_name = "./after_complete_2hop_kb_qa.txt"

    if it is 3hop, use:
        kgc_file = "./after_complete_3hop_kb.txt"
        file_name = "./after_complete_3hop_kb_qa.txt"
    """
    kgc_file = "./after_complete_2hop_kb.txt"
    file_name = "./after_complete_2hop_kb_qa.txt"

    kgc_triple = []
    f = open(kgc_file,'r')
    for line in f:
        line = line.strip().split('\t')
        kgc_triple.append(line)
    f.close()

    kgc_map = {}
    for e in kgc_triple:
        h = e[0]
        r = e[1]
        t = e[2]
        if h not in kgc_map:
            kgc_map[h] = {}
        if r not in kgc_map[h]:
            kgc_map[h][r] = set()
        kgc_map[h][r].add(t)

    kgc_to_qa_list = []
    for h in kgc_map.keys():
        for r in kgc_map[h].keys():
            ans = list(kgc_map[h][r])
            if not split:
                joined_string = "|".join(ans)
                qa_tt = "[" + h + "] " + r
                kgc_to_qa_list.append([qa_tt, joined_string, r])
            else:
                for a_ in ans:
                    qa_tt = "[" + h + "] " + r
                    kgc_to_qa_list.append([qa_tt, a_, r])
    
    #write 
    

    with open(file_name, 'w') as fp:
        for e in kgc_to_qa_list:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\n")
    fp.close()

    # merge them
    a = 0
        


if __name__ == '__main__':
    merge_kgc_with_QA_old(split=False)
