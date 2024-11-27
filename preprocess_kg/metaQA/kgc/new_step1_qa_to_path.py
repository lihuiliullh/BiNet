import os
import pickle
import argparse
import json
from collections import defaultdict




def read_kg(kg_file):
    kg_triple = []
    f = open(kg_file,'r')
    for line in f:
        line = line.strip().split('\t')
        kg_triple.append(line)
    f.close()
    return kg_triple



def process_text_file(text_file, split=True):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        if "noop" in data_line:
            continue
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        if len(data_line) < 2:
            print(data_line)
            continue
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data


def analyze_2hop():
    # this is ok to use any kg. because it is used to generate path
    full_triple_file = "./data/MetaQA_full/train.txt"
    f = open(full_triple_file,'r')
    full_training_triples = []
    for line in f:
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        full_training_triples.append(parts)
        # add reverse
        h = parts[2]
        r = parts[1] + "_reverse"
        t = parts[0]
        full_training_triples.append([h, r, t])
    f.close()

    adj_map = {}
    for e in full_training_triples:
        h = e[0]
        r = e[1]
        t = e[2]
        if h not in adj_map:
            adj_map[h] = {}
        if r not in adj_map[h]:
            adj_map[h][r] =  set()
        adj_map[h][r].add(t)


    # qa data are all the same, the only difference is the kg
    qa_file = "./data/QA_data/MetaQA/old/qa_train_2hop_old.txt"

    qa_triple = []
    f = open(qa_file,'r')
    for line in f:
        line = line.strip()

        parts = line.split("\t")
        eles = parts[0].split("[")
        _ = eles[1].split("]")
        head = _[0]
        qs = eles[0] + "NE" + _[1]
        ans = parts[1].split("|")
        qa_triple.append([head, qs, set(ans)])
    f.close()


    # iterate qa and find path
    qa_to_relation_path_adj = {}
    for e in qa_triple:
        h = e[0]
        txt = e[1]
        ans_set = e[2]
        # iterate map
        one_hop = adj_map[h]
        for rel, neis in one_hop.items():
            for node in neis:
                two_hop = adj_map[node]
                for rel2, neis2 in two_hop.items():
                    for n2 in neis2:
                        if n2 in ans_set:
                            # logic code
                            if txt not in qa_to_relation_path_adj:
                                qa_to_relation_path_adj[txt] = defaultdict(int)
                            qa_to_relation_path_adj[txt][rel + "|" + rel2] += 1


    # check whether all the query in map
    # one questions is not in qa_to_relation_path_adj
    #def check_qa_inside():
    for e in qa_triple:
        h = e[0]
        txt = e[1]
        ans_set = e[2]
        if txt not in qa_to_relation_path_adj:
            print(e)

    # only keep the path with the highest count
    qa_to_path_map = {}
    for k, v in qa_to_relation_path_adj.items():
        path = ""
        ct = 0
        for k_, v_ in v.items():
            if v_ > ct:
                ct = v_
                path = k_
        qa_to_path_map[k] = path
    # store file
    with open('qa_to_relation_path_2hop.pkl', 'wb') as f:
        pickle.dump(qa_to_path_map, f)



def analyze_1hop():
    full_triple_file = "./data/MetaQA_full/train.txt"
    f = open(full_triple_file,'r')
    full_training_triples = []
    for line in f:
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        full_training_triples.append(parts)
        # add reverse
        h = parts[2]
        r = parts[1] + "_reverse"
        t = parts[0]
        full_training_triples.append([h, r, t])
    f.close()

    adj_map = {}
    for e in full_training_triples:
        h = e[0]
        r = e[1]
        t = e[2]
        if h not in adj_map:
            adj_map[h] = {}
        if r not in adj_map[h]:
            adj_map[h][r] =  set()
        adj_map[h][r].add(t)

    # here should use qa_train_1hop.txt, because it contains triples
    # here can use qa_train_1hop.txt or qa_train_1hop_old.txt, the results should be the same
    qa_file = "./data/QA_data/MetaQA/old/qa_train_1hop.txt"

    all_question_type = set()
    qa_triple = []
    f = open(qa_file,'r')
    for line in f:
        line = line.strip()

        parts = line.split("\t")
        eles = parts[0].split("[")
        _ = eles[1].split("]")
        head = _[0]
        qs = eles[0] + "NE" + _[1]
        ans = parts[1].split("|")
        qa_triple.append([head, qs, set(ans)])
    f.close()

    # iterate qa and find path
    qa_to_relation_path_adj = {}
    for e in qa_triple:
        h = e[0]
        txt = e[1]
        ans_set = e[2]
        # iterate map
        one_hop = adj_map[h]
        for rel, neis in one_hop.items():
            for node in neis:
                if node in ans_set:
                    if txt not in qa_to_relation_path_adj:
                        qa_to_relation_path_adj[txt] = defaultdict(int)
                    qa_to_relation_path_adj[txt][rel] += 1
    

    # check whether all the query in map
    # one questions is not in qa_to_relation_path_adj
    def check_qa_inside():
        for e in qa_triple:
            h = e[0]
            txt = e[1]
            ans_set = e[2]
            if txt not in qa_to_relation_path_adj:
                print(e)
    


    # only keep the path with the highest count
    qa_to_path_map = {}
    for k, v in qa_to_relation_path_adj.items():
        path = ""
        ct = 0
        for k_, v_ in v.items():
            if v_ > ct:
                ct = v_
                path = k_
        qa_to_path_map[k] = path

    # store file
    with open('qa_to_relation_path_1hop.pkl', 'wb') as f:
        pickle.dump(qa_to_path_map, f)


def analyze_3hop():
    # to find path for qa
    # find all the shortest path (r1, r2, r3) between the anchor entity in QA and answers (like 1-hop and 2-hop)
    # only keep the path which r2 == r1_reverse or r1_reverse = r2
    # choose the path with the largest count
    file_name = "./preprocess/kgc/MetaQA_3hop_path.txt"
    qa_2_path_map = {}
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qa_2_path_map[parts[0]] = parts[2]
    
    f.close()
    
    # check whether all qa_test in map

    with open('qa_to_relation_path_3hop.pkl', 'wb') as f:
        pickle.dump(qa_2_path_map, f)
    a = 0



if __name__ == '__main__':

    # this corresponds to section section 3.2: how to find the groundtruth path "we choose the path with the highest probability as the answer."
    # only used for metaQA, because the path for webqsp and simpleQA already given in the dataset, do not need to infer
    analyze_1hop()
    analyze_2hop()
    analyze_3hop()
