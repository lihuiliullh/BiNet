import os
import pickle
import argparse
# check the accuracy
import json


"""
The generated after_complete_xxxxx_kb.txt contains all the triple in KG and all the complete triple

remember to change the file_path in the code:
    if you run the file for MetaQA half, change all kgc_file(variable) to MetaQA half
    if you run the file for MetaQA 30, change all kgc_file(variable) to MetaQA 30
"""

# this file only process metaQA 2hop and 3hop

def get_adjmap(kgc_triple):
    adj_map = {}
    for e in kgc_triple:
        h = e[0]
        r = e[1]
        t = e[2]
        if h not in adj_map:
            adj_map[h] = {}
        if r not in adj_map[h]:
            adj_map[h][r] = set()
        adj_map[h][r].add(t)
    return adj_map


def process_qa_file(qa_file):
    qa_data = []
    f = open(qa_file,'r')
    for line in f:
        line = line.strip()

        parts = line.split("\t")
        eles = parts[0].split("[")
        xx = eles[1].split("]")
        head = xx[0]
        qs = eles[0] + "NE" + xx[1]

        answer = parts[1].split("|")
        qa_data.append([head, qs, answer])
        a = 0
    f.close()
    return qa_data


def complete_2hop(qa_data, adj_map, qa_good_pattern_map):
    edges_find = []
    for e in qa_data:
        head = e[0]
        qs = e[1]
        answer = e[2]
        
        if qs not in qa_good_pattern_map:
            continue
        # find the answer
        pattern_of_qa = [qa_good_pattern_map[qs]]
        if len(pattern_of_qa) > 1:
            print("error")
        for pattern_candidate in pattern_of_qa:
            rels = pattern_candidate.split("|")
            # main logic
            if len(answer) == 1:
                # forward
                if rels[0] in adj_map[head]:
                    one_hop = adj_map[head][rels[0]]
                    if len(one_hop) == 1:
                        m = list(one_hop)[0]
                        if rels[1] not in adj_map[m]:
                            edges_find.append([m, rels[1], answer[0]])
                        else:
                            two_hop = adj_map[m][rels[1]]
                            if answer[0] not in two_hop:
                                edges_find.append([m, rels[1], answer[0]])
                else:
                    # backward
                    if "reverse" in rels[1]:
                        r2 = rels[1][0:-8]
                    else:
                        r2 = rels[1] + "_reverse"
                    one_hop = adj_map[answer[0]]
                    if r2 in one_hop:
                        one_hop_node = one_hop[r2]
                        if len(one_hop_node) == 1:
                            edges_find.append([head, rels[0], list(one_hop_node)[0]])
            else:
                aaa = 0
                """
                if rels[0] in adj_map[head]:
                    one_hop = adj_map[head][rels[0]]
                    if len(one_hop) == 1:
                        one_hop_node = list(one_hop)[0]
                        if one_hop_node == "First Family":
                            xxxxxxxxxx = 1
                        if rels[1] not in adj_map[one_hop_node]:
                            _ = 0
                            for a_ in answer:
                                edges_find.append([one_hop_node, rels[1], a_])
                        else:
                            xx = adj_map[one_hop_node][rels[1]]
                            for a_ in answer:
                                if a_ not in xx:
                                    edges_find.append([one_hop_node, rels[1], a_])
                else:
                    if "reverse" in rels[1]:
                        r2 = rels[1][0:-8]
                    else:
                        r2 = rels[1] + "_reverse"
                    res = None
                    for a_ in answer:
                        if r2 in adj_map[a_]:
                            if res is None:
                                res = adj_map[a_][r2]
                            else:
                                z = res.intersection(adj_map[a_][r2])
                    if res is not None and len(res) == 1:
                        edges_find.append([head, rels[0], list(res)[0]])
                """
    return edges_find


def complete_3hop(qa_data, adj_map, qa_good_pattern_map):
    edges_find = []
    for e in qa_data:
        head = e[0]
        qs = e[1]
        answer = e[2]
        
        if qs not in qa_good_pattern_map:
            continue
        # find the answer
        pattern_of_qa = [qa_good_pattern_map[qs]]
        if len(pattern_of_qa) > 1:
            print("error")
        for pattern_candidate in pattern_of_qa:
            rels = pattern_candidate.split("|")
            # main logic
            # forward
            if rels[0] in adj_map[head]:
                one_hop = adj_map[head][rels[0]]
                # find all two hop
                all_two = set()
                for one_nei in one_hop:
                    if rels[1] in adj_map[one_nei]:
                        all_two |= adj_map[one_nei][rels[1]]
                
                # delete itself
                all_two.discard(head)

                if len(all_two) == 1:
                    if len(answer) > 1:
                        continue
                    if rels[2] in adj_map[list(all_two)[0]]:
                        for x_ in answer:
                            if x_ not in adj_map[list(all_two)[0]][rels[2]]:
                                edges_find.append([list(all_two)[0], rels[2], x_])
                    else:
                        for x_ in answer:
                            edges_find.append([list(all_two)[0], rels[2], x_])
            else:
                aaa = 0
            
            # backward
            if rels[0] not in adj_map[head]:
                # reverse_relation
                if "reverse" in rels[2]:
                    r2 = rels[2][0:-8]
                else:
                    r2 = rels[2] + "_reverse"
                if "reverse" in rels[1]:
                    r1 = rels[1][0:-8]
                else:
                    r1 = rels[1] + "_reverse"
                
                # iterate
                candicate_set = set()
                x_set = set()
                for a_ in answer:
                    if r2 in adj_map[a_]:
                        x_set |= adj_map[a_][r2]
                for a_ in x_set:
                    if r1 in adj_map[a_]:
                        candicate_set |= adj_map[a_][r1]
                
                if len(candicate_set) == 1:
                    edges_find.append([head, rels[0], list(candicate_set)[0]])
            
            # middle
            if rels[0] in adj_map[head]:
                if "reverse" in rels[2]:
                    r2 = rels[2][0:-8]
                else:
                    r2 = rels[2] + "_reverse"
                
                x_set = set()
                for a in answer:
                    if r2 in adj_map[a]:
                        x_set |= adj_map[a][r2]
                
                if len(x_set) == 1:
                    t_ = adj_map[head][rels[0]]
                    # none of t connect to x_set
                    None_connect = True
                    for n_ in t_:
                        if rels[1] in adj_map[n_] and list(x_set)[0] in adj_map[n_][rels[1]]:
                            None_connect = False
                    if None_connect:
                        for n_ in t_:
                            edges_find.append([n_, rels[1], list(x_set)[0]])
            

    return edges_find


def complete_2hop_with_good_pattern(task_type):
    # read good pattern
    good_pattern_file = "./qa_to_relation_path_2hop.pkl"
    with open(good_pattern_file, 'rb') as f:
        qa_good_pattern_map = pickle.load(f)

    # build adj map
    if task_type == 'half':
        kgc_file = "./data/MetaQA_half/train.txt"
    elif task_type == '30':
        kgc_file = "./data/MetaQA_30/train.txt"
    else:
        print("error, task type is wrong")
        return

    kgc_triple = []
    f = open(kgc_file,'r')
    for line in f:
        line = line.strip().split('\t')
        # don't skip noop
        # it needs to be a full kg
        #if line[1] == "noop":
        #    continue
        kgc_triple.append(line)
        # add reverse
        kgc_triple.append([line[2], line[1] + "_reverse", line[0]])
    f.close()


    adj_map = get_adjmap(kgc_triple)
    


    qa_file = "./data/QA_data/MetaQA/old/qa_train_2hop_old.txt"
    qa_data = []
    f = open(qa_file,'r')
    for line in f:
        line = line.strip()

        parts = line.split("\t")
        eles = parts[0].split("[")
        xx = eles[1].split("]")
        head = xx[0]
        qs = eles[0] + "NE" + xx[1]

        answer = parts[1].split("|")
        qa_data.append([head, qs, answer])
        a = 0
    f.close()


    qa_file = "./data/QA_data/MetaQA/qa_dev_2hop.txt"
    f = open(qa_file,'r')
    for line in f:
        line = line.strip()

        parts = line.split("\t")
        eles = parts[0].split("[")
        xx = eles[1].split("]")
        head = xx[0]
        qs = eles[0] + "NE" + xx[1]

        answer = parts[1].split("|")
        qa_data.append([head, qs, answer])
        a = 0
    f.close()



    while True:
        edges_find = complete_2hop(qa_data, adj_map, qa_good_pattern_map)

        if len(edges_find) == 0:
            break
        
        deduplicate = {}
        really_good_results = []
        for e in edges_find:
            if "reverse" in e[1]:
                r2 = e[1][0:-8]
                key = (e[2], r2, e[0])
                if key not in deduplicate:
                    really_good_results.append([e[2], r2, e[0]])
                    deduplicate[key] = 1
            else:
                key = (e[0], e[1], e[2])
                if key not in deduplicate:
                    really_good_results.append([e[0], e[1], e[2]])
                    deduplicate[key] = 1
        
        # merge really_good_results
        # add reverse to really_good_results
        really_good_results_reverse = []
        for e in really_good_results:
            really_good_results_reverse.append([e[2], e[1] + "_reverse", e[0]])
        
        kgc_triple.extend(really_good_results)
        kgc_triple.extend(really_good_results_reverse)

        adj_map = get_adjmap(kgc_triple)
    

    # if node connected with others, but has noop
    # delete noop, and delete reverse
    noop_map = {}
    other_triples = []
    other_nodes = {}
    for e in kgc_triple:
        if e[1].startswith("noop"):
            if "reverse" in e[1]:
                continue
            noop_map[e[0]] = e
        else:
            # why does not keep "reverse"?
            if "reverse" in e[1]:
                continue
            other_triples.append(e)
            other_nodes[e[0]] = 1
            other_nodes[e[2]] = 1
    
    for k, v in noop_map.items():
        if k in other_nodes:
            continue
        else:
            other_triples.append(v)

    
    
    file_name = "after_complete_2hop_kb.txt"

    with open(file_name, 'w') as fp:
        for e in other_triples:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    fp.close()

    # merge them
    a = 0



def complete_3hop_with_good_pattern(task_type):
    # read good pattern
    good_pattern_file = "./qa_to_relation_path_3hop.pkl"
    with open(good_pattern_file, 'rb') as f:
        qa_good_pattern_map = pickle.load(f)

    # build adj map
    if task_type == 'half':
        kgc_file = "./data/MetaQA_half/train.txt"
    elif task_type == '30':
        kgc_file = "./data/MetaQA_30/train.txt"
    else:
        print("error, task type is wrong")
        return 
    #kgc_file = "merged_triples_with_train_iter4.txt"

    kgc_triple = []
    f = open(kgc_file,'r')
    for line in f:
        line = line.strip().split('\t')
        # don't skip noop
        # it needs to be a full kg
        #if line[1] == "noop":
        #    continue
        kgc_triple.append(line)
        # add reverse
        kgc_triple.append([line[2], line[1] + "_reverse", line[0]])
    f.close()


    adj_map = get_adjmap(kgc_triple)
    


    qa_file = "./data/QA_data/MetaQA/old/qa_train_3hop_old.txt"
    qa_data = process_qa_file(qa_file)
    
    qa_file = "./data/QA_data/MetaQA/qa_dev_3hop.txt"
    qa_data2 = process_qa_file(qa_file)
    qa_data.extend(qa_data2)

    while True:
        edges_find = complete_3hop(qa_data, adj_map, qa_good_pattern_map)

        if len(edges_find) == 0:
            break
        
        deduplicate = {}
        really_good_results = []
        for e in edges_find:
            if "reverse" in e[1]:
                r2 = e[1][0:-8]
                key = (e[2], r2, e[0])
                if key not in deduplicate:
                    really_good_results.append([e[2], r2, e[0]])
                    deduplicate[key] = 1
            else:
                key = (e[0], e[1], e[2])
                if key not in deduplicate:
                    really_good_results.append([e[0], e[1], e[2]])
                    deduplicate[key] = 1
        
        # merge really_good_results
        # add reverse to really_good_results
        really_good_results_reverse = []
        for e in really_good_results:
            really_good_results_reverse.append([e[2], e[1] + "_reverse", e[0]])
        
        kgc_triple.extend(really_good_results)
        kgc_triple.extend(really_good_results_reverse)

        adj_map = get_adjmap(kgc_triple)
    

    # if node connected with others, but has noop
    # delete noop, and delete reverse
    noop_map = {}
    other_triples = []
    other_nodes = {}
    for e in kgc_triple:
        if e[1].startswith("noop"):
            if "reverse" in e[1]:
                continue
            noop_map[e[0]] = e
        else:
            if "reverse" in e[1]:
                continue
            other_triples.append(e)
            other_nodes[e[0]] = 1
            other_nodes[e[2]] = 1
    
    for k, v in noop_map.items():
        if k in other_nodes:
            continue
        else:
            other_triples.append(v)

    
    
    file_name = "after_complete_3hop_kb.txt"

    with open(file_name, 'w') as fp:
        for e in other_triples:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    fp.close()

    # merge them
    a = 0 


if __name__ == '__main__':
    """
    The generated after_complete_xxxxx_kb.txt contains all the triple in KG and all the complete triple

    remember to change the file_path in the code:
        if you run the file for MetaQA half, change all kgc_file(variable) to MetaQA half
        if you run the file for MetaQA 30, change all kgc_file(variable) to MetaQA 30
    """
    task_type = "half"
    # task_type = "30"
    complete_2hop_with_good_pattern(task_type)
    complete_3hop_with_good_pattern(task_type)

