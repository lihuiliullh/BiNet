import os
import pickle
import json

# before run this file
# KGC is not used in 1hop query

# 1. copy qa_train_xxhop_old.txt and rename it to qa_train_xxhop_kgc.txt
# 2. copy the corresponding after_complete_xxx_kb_qa.txt to the end of qa_train_xxhop_kgc.txt (append it in the end)

# this file process metaQA 1hp, 2hop and 3hop

def add_path(qa_file_name, hop2_path_map, hop1_path_map):
    qa_path = []
    f = open(qa_file_name,'r')
    for line in f:
        line = line.strip()
        if line == '':
            continue
        parts = line.split("\t")
        eles = parts[0].split("[")
        _ = eles[1].split("]")
        head = _[0]
        qs = eles[0] + "NE" + _[1]
        ans = parts[1].split("|")

        if qs in hop2_path_map:
            path_ = hop2_path_map[qs]
        elif qs in hop1_path_map:
            path_ = hop1_path_map[qs]
            path_ = path_ + "|noop"
        else:
            path_ = "noop|noop"
        
        qa_path.append([line, path_])
    f.close()
    return qa_path


def add_path3(qa_file_name, hop2_path_map, hop1_path_map):
    qa_path = []
    f = open(qa_file_name,'r')
    for line in f:
        line = line.strip()
        if line == '':
            continue
        parts = line.split("\t")
        eles = parts[0].split("[")
        _ = eles[1].split("]")
        head = _[0]
        qs = eles[0] + "NE" + _[1]
        ans = parts[1].split("|")

        if qs in hop2_path_map:
            path_ = hop2_path_map[qs]
        elif qs in hop1_path_map:
            path_ = hop1_path_map[qs]
            path_ = path_ + "|noop|noop"
        else:
            path_ = "noop|noop|noop"
        
        qa_path.append([line, path_])
    f.close()
    return qa_path


def add_path1(qa_file_name, hop2_path_map, hop1_path_map):
    qa_path = []
    f = open(qa_file_name,'r')
    for line in f:
        line = line.strip()
        if line == '':
            continue
        parts = line.split("\t")
        eles = parts[0].split("[")
        _ = eles[1].split("]")
        head = _[0]
        qs = eles[0] + "NE" + _[1]
        ans = parts[1].split("|")

        if qs in hop2_path_map:
            path_ = hop2_path_map[qs]
        elif qs in hop1_path_map:
            path_ = hop1_path_map[qs]
            #path_ = path_ + "|noop|noop"
        else:
            path_ = "noop"
        
        qa_path.append([line, path_])
    f.close()
    return qa_path


def add_path_to_qa2():
    hop2_origin_qa_to_path_map = pickle.load(open( "qa_to_relation_path_2hop.pkl", "rb" ))
    hop1_origin_qa_to_path_map = pickle.load(open( "qa_to_relation_path_1hop.pkl", "rb" ))
    
    qa_file = "./data/QA_data/MetaQA/qa_train_2hop_half_kgc.txt"

    res1 = add_path(qa_file, hop2_origin_qa_to_path_map, hop1_origin_qa_to_path_map)

    # write new file
    file_name = "./data/QA_data/MetaQA/qa_train_2hop_half_kgc_path.txt"
    qa_path = res1
    with open(file_name, 'w') as fp:
        for e in qa_path:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\n")
    fp.close()

    a = 0


def add_path_to_qa3():
    hop2_origin_qa_to_path_map = pickle.load(open( "qa_to_relation_path_3hop.pkl", "rb" ))
    hop1_origin_qa_to_path_map = pickle.load(open( "qa_to_relation_path_1hop.pkl", "rb" ))
    
    qa_file = "./data/QA_data/MetaQA/qa_train_3hop_half_kgc.txt"

    res1 = add_path3(qa_file, hop2_origin_qa_to_path_map, hop1_origin_qa_to_path_map)

    # write new file
    file_name = "./data/QA_data/MetaQA/qa_train_3hop_half_kgc_path.txt"
    qa_path = res1
    with open(file_name, 'w') as fp:
        for e in qa_path:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\n")
    fp.close()

    a = 0



def add_path_to_qa1():
    hop1_origin_qa_to_path_map = pickle.load(open( "qa_to_relation_path_1hop.pkl", "rb" ))
    
    # this file is directly adopted from EmbedKGQA
    qa_file = "./data/QA_data/MetaQA/old/qa_train_1hop_half.txt"

    res1 = add_path1(qa_file, hop1_origin_qa_to_path_map, hop1_origin_qa_to_path_map)

    # write new file
    # there is no KGC here, but we need to keep their name consistent
    file_name = "./data/QA_data/MetaQA/qa_train_1hop_half_kgc_path.txt"
    qa_path = res1
    with open(file_name, 'w') as fp:
        for e in qa_path:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\n")
    fp.close()

    a = 0

if __name__ == '__main__':
    add_path_to_qa1()
    add_path_to_qa2()
    add_path_to_qa3()
