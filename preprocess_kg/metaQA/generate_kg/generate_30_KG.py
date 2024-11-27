
import random


def generate_30_no_kgc_for_MetaQA():
    # generate 1 hop
    # read half train.xt
    file_name = "./data/MetaQA_half/train.txt"
    good_triple = []
    bad_triple = []
    with open(file_name) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if parts[1] == "noop":
                bad_triple.append(parts)
            else:
                good_triple.append(parts)
    f.close()

    random.shuffle(good_triple)

    # choose 60% edges 
    keep_triple = good_triple[0: int(len(good_triple) * 0.6)]

    # exist_node
    exist_node_map = {}
    for e in keep_triple:
        exist_node_map[e[0]] = 1
        exist_node_map[e[2]] = 1
    
    deleted_node_map = {}
    for e in good_triple[int(len(good_triple) * 0.6) : -1]:
        if e[0] not in exist_node_map:
            deleted_node_map[e[0]] = 1
        if e[2] not in exist_node_map:
            deleted_node_map[e[2]] = 1
    
    output_file = "train.txt"
    with open(output_file, 'w') as f:
        for e in keep_triple:
            f.write(e[0] + "\t" + e[1] + "\t" + e[2] + "\n")
        
        for k in deleted_node_map:
            f.write(k + "\t" + "noop" + "\t" + k + "\n")
        
        for e in bad_triple:
            if e[0] in deleted_node_map:
                continue
            f.write(e[0] + "\t" + e[1] + "\t" + e[2] + "\n")
    
    f.close()




# fbwq_30 and fbsimpleQA_30 are generated in the same way
generate_30_no_kgc_for_MetaQA()

