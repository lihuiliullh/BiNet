import os
from numpy.core.numeric import False_
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
sys.path.append(".") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from knowledge_graph_completion import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

parser = argparse.ArgumentParser()


parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--load_from', type=str, default='')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--validate_every', type=int, default=5)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--outfile', type=str, default='best_score_model')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=90)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--relation_dim', type=int, default=30)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--freeze', type=str2bool, default=True)
parser.add_argument('--do_batch_norm', type=str2bool, default=True)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
            sent = d[1]
            for word in sent.split():
                if word not in word_to_ix:
                    idx2word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                    
            length = len(sent.split())
            if length > maxLength:
                maxLength = length

    return word_to_ix, idx2word, maxLength

def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict,'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e,r

def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('|')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G

def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key
    

def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes


def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x

def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result


# tmp_adj.pkl stores the information of the corresponding knowledge graph
tmp_file = "./tmp_adj.pkl"
with open(tmp_file, 'rb') as f:
    g_train_map = pickle.load(f)

def validate_v2(data_path, device, model, dataloader, entity2idx, relation2idx, model_name, writeCandidatesToFile=False):
    model.eval()
    data = process_text_file(data_path, relation2idx)
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = []
    data_gen = data_generator(data=data, dataloader=dataloader, entity2idx=entity2idx, relation2idx=relation2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0

    # print('Loading nbhood file')
    # if 'webqsp' in data_path:
    #     # with open('webqsp_test_candidate_list_only2hop.pickle', 'rb') as handle:
    #     with open('webqsp_test_candidate_list_half.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)
    # else:
    #     with open('qa_dev_full_candidate_list.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)

    id2entity_ = {}
    for k, v in entity2idx.items():
        id2entity_[v] = k
    
    id2relation_ = {}
    for k, v in relation2idx.items():
        id2relation_[v] = k

    scores_list = []
    hit_at_10 = 0
    total_correct_adj = 0
    total_correct_reasoning = 0
    candidates_with_scores = []
    answers_reasoning = []
    infos = []
    for i in tqdm(range(len(data))):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        rels = d[5].to(device)
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        top_2 = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask, p_rels=rels)
        top_2_idx = top_2[0][1].tolist()[0]
        head_idx = head.tolist()
        one_score = 0
        if top_2_idx[0] == head_idx:
            pred_ans = top_2_idx[1]
            one_score = top_2[0][0].tolist()[0][1]
        else:
            pred_ans = top_2_idx[0]
            one_score = top_2[0][0].tolist()[0][0]
        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        q_text = d[-2]
        answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))

        #######################################
        two_score = 0
        three_score = 0
        top_2_idx_reasoning = top_2[1][1].tolist()[0]
        if top_2_idx_reasoning[0] == head_idx:
            pred_ans_transformer = top_2_idx_reasoning[1]
            two_score = top_2[1][0].tolist()[0][1]
            three_score = top_2[1][0].tolist()[0][0]
        else:
            pred_ans_transformer = top_2_idx_reasoning[0]
            two_score = top_2[1][0].tolist()[0][0]
            three_score = top_2[1][0].tolist()[0][1]
        is_correct_reasoning = 0
        if pred_ans_transformer in ans:
            total_correct_reasoning += 1
            is_correct_reasoning = 1
        q_text = d[-2]
        answers_reasoning.append(q_text + '\t' + str(pred_ans_transformer) + '\t' + str(is_correct_reasoning))

        # reranking model
        # if pred_ans is different from pred_ans_transformer
        # find answer depends on which one is connected to head in kG
        if pred_ans != pred_ans_transformer:
            h = id2entity_[head_idx]
            rs = rels.tolist()
            candidate_answers = []
            for r_ in rs:
                r_v = id2relation_[r_]
                key = (h, r_v)
                if key in g_train_map:
                    candidate_answers.extend(list(g_train_map[key]))
            candidate_answers = set(candidate_answers)
            if id2entity_[pred_ans] in candidate_answers:
                adj_pred_ans = pred_ans
            else:
                adj_pred_ans = pred_ans_transformer
            
            if adj_pred_ans in ans:
                total_correct_adj += 1
        else:
            if is_correct_reasoning == 1:
                total_correct_adj += 1
    
    accuracy = total_correct/len(data)
    accuracy_reasoning = total_correct_reasoning/len(data)
    print("BiNet ACC")
    print(total_correct_adj / len(data))
    return answers, accuracy, answers_reasoning, accuracy_reasoning





def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(kge_model, hops):
    e = {}
    entity_dict = './pretrained_models/embeddings/ComplEx_fbwq_full/entity_ids.del'
    if 'half' in hops:
        entity_dict = './pretrained_models/embeddings/ComplEx_fbwq_half/entity_ids.del'
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e

def getRelationEmbeddings(kge_model, hops):
    e = {}
    relation_dict = './data/fbwq_full/relation_ids.del'
    if 'half' in hops:
        relation_dict = './data/fbwq_half/relation_ids.del'
        print('Loading half relation_ids.del')
    embedder = kge_model._relation_embedder
    f = open(relation_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        e[rel_name] = embedder._embeddings(torch.LongTensor([rel_id]))[0]
    f.close()
    return e

def train(data_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, load_from, outfile, do_batch_norm, valid_data_path=None):
    print('Loading entities and relations')
    kg_type = 'full'
    if 'half' in hops:
        kg_type = 'half'
    checkpoint_file = './pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
    print('Loading kg embeddings from', checkpoint_file)
    kge_checkpoint = load_checkpoint(checkpoint_file)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()
    e = getEntityEmbeddings(kge_model, hops)
    r = getRelationEmbeddings(kge_model, hops)

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    relation2idx, idx2relation, rel_embedding_matrix = prepare_embeddings(r)

    data = process_text_file(data_path, relation2idx, split=False)
    print('Train file processed, making dataloader')
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, e, entity2idx, relation2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, 
        pretrained_node_embeddings=embedding_matrix,  pretrained_relation_embeddings=rel_embedding_matrix,
        freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, do_batch_norm=do_batch_norm)
    print('Model created!')
    
    # load model
    #fname = "checkpoints/roberta_finetune/half_fbwq.pt"
    # fname = "/scratche/home/apoorv/tut_pytorch/kg-qa/checkpoints/roberta_finetune/" + load_from + ".pt"
    #model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')), strict=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    # time.sleep(10)

    for epoch in range(nb_epochs):
        phases = []
        #phases.append('valid')
        #for i in range(1):
        #validate_every = 5
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                # model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2].to(device)
                    positive_tail = a[3].to(device)    
                    positive_rels = a[4].to(device)
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail, p_rels=positive_rels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()

                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                answers, score, answer_rs, score_rs = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx, relation2idx=relation2idx, dataloader=dataset, device=device, model_name=model_name)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
                    print('Relation Reasoning: Test score for best valid so far:', score_rs)
                    # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")

                     # do kb test here
                    __, kb_test_score = kb_test('./data/fbwq_half/test.txt', device, model, entity2idx, relation2idx)
                    print('Test knowledge graph completion score for best valid so far:', kb_test_score)

                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/"+str(epoch)+".pt")
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/x.pt")

def eval(data_path,
    load_from,
    gpu,
    hidden_dim,
    relation_dim,
    embedding_dim,
    hops,
    batch_size,
    num_workers,
    model_name,
    do_batch_norm,
    use_cuda):

    print('Loading entities and relations')
    kg_type = 'full'
    if 'half' in hops:
        kg_type = 'half'
    checkpoint_file = './pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
    print('Loading kg embeddings from', checkpoint_file)
    kge_checkpoint = load_checkpoint(checkpoint_file)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()
    e = getEntityEmbeddings(kge_model, hops)

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data = process_text_file(data_path, split=False)
    print('Evaluation file processed, making dataloader')

    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, e, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, 
                              pretrained_node_embeddings=embedding_matrix, device=device, 
                              model = model_name, do_batch_norm=do_batch_norm)
    print('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        print('Loading from %s' % fname)
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        print('Loaded successfully!')
    else:
        print('Need to specify load_from argument for evaluation!')
        exit(0)
    
    model.to(device)
    answers, score = validate_v2(model=model, data_path= data_path, 
                                 entity2idx=entity2idx, dataloader=dataset, 
                                 device=device, model_name=model_name,
                                 writeCandidatesToFile=True)
    print('Score', score)


def process_text_file(text_file, relation2idx, split=False):
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
        for e__ in rels:
            if e__ in relation2idx:
                at_least_one_in = True
                break
        if not at_least_one_in:
            print(data_line)
            continue

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

def data_generator(data, dataloader, entity2idx, relation2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            #TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]
        rel_ids = []
        for rel_name in data_sample[3]:
            rel_name = rel_name.strip()
            if rel_name in relation2idx:
                rel_ids.append(relation2idx[rel_name])
            else:
                error = 0
        rel_score = sampleRelation(rel_ids)

        yield torch.tensor(head, dtype=torch.long), question_tokenized, attention_mask, ans, data_sample[1], rel_score

def relToOneHot(indices, relation2idx):
    num_rel = len(indices)
    indices = torch.LongTensor(indices)
    vec_len = len(relation2idx)
    one_hot = torch.FloatTensor(vec_len)
    one_hot.zero_()
    # one_hot = -torch.ones(vec_len, dtype=torch.float32)
    one_hot.scatter_(0, indices, 1 / num_rel)
    return one_hot

def sampleRelation(indices):
    # sample 4 indices here
    sample_size = 1
    if len(indices) >= sample_size:
        indices = indices[0:sample_size]
        sampled_neighbors = indices
    else:
        lef_len = sample_size - len(indices)
        tmp = np.random.choice(list(indices), size=lef_len, replace=len(indices) < sample_size)
        indices.extend(tmp)
        sampled_neighbors = indices
    
    return torch.LongTensor(sampled_neighbors)


################
# before running the code, chance line 270 tmp_adj.pkl

hops = args.hops

model_name = args.model

if 'webqsp' in hops:
    data_path = './data/QA_data/WebQuestionsSP/qa_train_webqsp_path.txt'
    valid_data_path = './data/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_path.txt'
    test_data_path = './data/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_path.txt'
    print(test_data_path)


if 'simpleqa' in hops:
    data_path = './data/QA_data/SimpleQA/qa_train_simpleqa_path.txt'
    valid_data_path = './data/QA_data/SimpleQA/qa_test_simpleqa_path.txt'
    test_data_path = './data/QA_data/SimpleQA/qa_test_simpleqa_path.txt'
    print(test_data_path)


if args.mode == 'train':
    train(data_path=data_path, 
    neg_batch_size=args.neg_batch_size, 
    batch_size=args.batch_size,
    shuffle=args.shuffle_data, 
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
    relation_dim=args.relation_dim, 
    gpu=args.gpu, 
    use_cuda=args.use_cuda, 
    valid_data_path=valid_data_path,
    patience=args.patience,
    validate_every=args.validate_every,
    freeze=args.freeze,
    hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    load_from=args.load_from,
    outfile=args.outfile,
    do_batch_norm=args.do_batch_norm)



elif args.mode == 'eval':
    eval(data_path = test_data_path,
    load_from=args.load_from,
    gpu=args.gpu,
    hidden_dim=args.hidden_dim,
    relation_dim=args.relation_dim,
    embedding_dim=args.embedding_dim,
    hops=args.hops,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    model_name=args.model,
    do_batch_norm=args.do_batch_norm,
    use_cuda=args.use_cuda)