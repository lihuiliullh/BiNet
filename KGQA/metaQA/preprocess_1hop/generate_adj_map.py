import pickle


def gegerate_adj_map():
    kg_type = "half"

    if kg_type == "half":
        kg_path = "./data/MetaQA_half/train.txt"
        out_file = "./MetaQA1hop_half_kgc_adj.pkl"
    elif kg_type == "30":
        kg_path = "./data/MetaQA_30/train.txt"
        out_file = "./MetaQA1hop_half_30_adj.pkl"
    else:
        kg_path = "./data/MetaQA_full/train.txt"

    # key : (head, relation). Value: set(tail_nodes)
    adj_map = {}
    f = open(kg_path,'r')
    full_training_triples = []
    for line in f:
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        
        key = (parts[0], parts[1])
        if key not in adj_map:
            adj_map[key] = set()
        
        adj_map[key].add(parts[2])

        # add reverse
        h = parts[2]
        r = parts[1] + "_reverse"
        t = parts[0]
        key = (h, r)
        if key not in adj_map:
            adj_map[key] = set()
        
        adj_map[key].add(t)
    f.close()

    # read qa file
    qa_file = "./data/QA_data/MetaQA/qa_train_1hop_half_kgc_path.txt"

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
        r = parts[2].strip()
        key = (head, r)
        if key not in adj_map:
            adj_map[key] = set()
        for t in ans:
            adj_map[key].add(t)
        
    f.close()

    # write adj_map to pickle
    with open(out_file, 'wb') as file:
        pickle.dump(adj_map, file)
    
gegerate_adj_map()