import random
from os.path import exists
import pickle


def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 3:
            continue
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        ans = data_line[1].split('|')
        rels = data_line[2].split("|")
        at_least_one_in = False
        data_array.append([head, question.strip(), ans, rels])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            rels = line[3]
            for tail in tails:
                data.append([head, question, tail, rels])
        return data



def analyze_adj():
    train_txt = "./data/fbwq_half/train.txt"

    tmp_file = "./tmp_adj.pkl"
    if exists(tmp_file):
        with open(tmp_file, 'rb') as f:
            train_map = pickle.load(f)
    else:
        train_map = {}
        with open(train_txt, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = parts[0]
                r = parts[1]
                t = parts[2]
                key = (h, r)
                if key not in train_map:
                    train_map[key] = set()
                train_map[key].add(t)
        # read training
        train_qa = "./data/QA_data/WebQuestionsSP/qa_train_webqsp_path.txt"
        train_info = process_text_file(train_qa)
        for e_ in train_info:
            h = e_[0]
            rels = e_[3]
            ans = e_[2]
            for r in rels:
                key = (h, r)
                if key not in train_map:
                    train_map[key] = set()
                for a_ in ans:
                    train_map[key].add(a_)
        
        with open('./tmp_adj.pkl', 'wb') as f:
            pickle.dump(train_map, f)
    
    test_qa = "./data/QA_data/WebQuestionsSP/qa_test_webqsp_path.txt"
    test_info = process_text_file(test_qa)
    no_answer = []
    for e in test_info:
        h = e[0]
        ans = e[2]
        rels = e[3]
        # check whether in train
        Find = False
        for r in rels:
            key = (h, r)
            if key in train_map:
                # check answer
                cands = train_map[key]
                for a_ in ans:
                    if a_ in cands:
                        Find = True
            if Find:
                break
        if not Find:
            no_answer.append(e)
    
    print(1 - len(no_answer) / len(test_info))

    a = 0




analyze_adj()
