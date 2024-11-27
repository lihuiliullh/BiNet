def add_path_to_train_query():
    data_query = "./KGQA/webqsp_simpleqa/pruning_train.txt"

    half_query = "./data/QA_data/WebQuestionsSP/qa_train_webqsp.txt"

    query_2_path_map = {}
    with open(data_query, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k = parts[0]
            v = parts[1]
            if k not in query_2_path_map:
                query_2_path_map[k] = set()
            query_2_path_map[k].add(v)
    
    res_triple = []
    
    with open(half_query, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                print(line)
                continue
            k = parts[0]
            if k not in query_2_path_map:
                continue
            v = query_2_path_map[k]
            assert len(v) == 1
            if len(v) > 1:
                error = 1

            res_triple.append([parts[0].strip(), parts[1].strip(), list(v)[0].strip()])
    
    f.close()

    # write to file
    output_file = "./data/QA_data/WebQuestionsSP/qa_train_webqsp_path.txt"

    with open(output_file, 'w') as fp:
        for e in res_triple:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    fp.close()

    a = 0




def add_path_to_test_query():
    data_query = "./KGQA/webqsp_simpleqa/pruning_test.txt"

    half_query = "./data/QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt"

    query_2_path_map = {}
    with open(data_query, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k = parts[0]
            v = parts[1]
            if k not in query_2_path_map:
                query_2_path_map[k] = set()
            query_2_path_map[k].add(v)
    
    res_triple = []
    
    with open(half_query, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                print(line)
                continue
            k = parts[0]
            if k not in query_2_path_map:
                continue
            v = query_2_path_map[k]
            assert len(v) == 1
            if len(v) > 1:
                error = 1

            res_triple.append([parts[0].strip(), parts[1].strip(), list(v)[0].strip()])
    
    f.close()

    # write to file
    output_file = "./data/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_path.txt"

    with open(output_file, 'w') as fp:
        for e in res_triple:
            fp.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    fp.close()

    a = 0



add_path_to_train_query()
add_path_to_test_query()
