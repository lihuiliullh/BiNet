import torch
import traceback
from tqdm import tqdm

def read_kg(kg_file):
    kg_triple = []
    f = open(kg_file,'r')
    for line in f:
        line = line.strip().split('\t')
        if len(line) < 2:
            continue
        kg_triple.append(line)
    f.close()
    return kg_triple


def kb_test_data_generator(data, entity2idx, relation2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        tail = entity2idx[data_sample[2].strip()]
        rel = relation2idx[data_sample[1].strip()]
        yield torch.tensor([head], dtype=torch.long), torch.tensor([rel], dtype=torch.long), torch.tensor([tail], dtype=torch.long)




def kb_test(data_path, device, model, entity2idx, relation2idx):
    model.eval()

    # process data
    test_triple = read_kg(data_path)

    answers = []
    data_gen = kb_test_data_generator(data=test_triple, entity2idx=entity2idx, relation2idx = relation2idx)
    total_correct = 0
    error_count = 0
    for i in tqdm(range(len(test_triple))):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            rel = d[1].to(device)
            ans = d[2]
            top_2 = model.get_kb_test_score_ranked(head=head, rel=rel)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]
            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1
            answers.append(str(pred_ans) + '\t' + str(is_correct))
        except:
            error_count += 1
            traceback.print_exc()

    print(error_count)
    accuracy = total_correct/len(test_triple)
    return answers, accuracy

