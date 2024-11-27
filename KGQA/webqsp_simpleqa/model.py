import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import *
from transformers import RobertaModel
import random
from transformer import *



class ThinMulticlassClassification(nn.Module):
    def __init__(self, input_dim):
        super(ThinMulticlassClassification, self).__init__()

        #self.layer_1 = nn.Linear(num_feature, num_feature)
        self.layer_2 = nn.Linear(input_dim, int(input_dim * 0.5))
        self.layer_3 = nn.Linear(int(input_dim * 0.5), int(input_dim * 0.5))
        self.layer_out = nn.Linear(int(input_dim * 0.5), 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm2 = nn.BatchNorm1d(int(input_dim * 0.5))
        self.batchnorm3 = nn.BatchNorm1d(int(input_dim * 0.5))
        self.m = nn.Sigmoid()

    def forward(self, x):

        x = self.layer_2(x)
        #x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        #x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        #x = self.m(x)

        return x



class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, relation_dim, num_entities, pretrained_node_embeddings, pretrained_relation_embeddings, device, 
    entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, model='ComplEx', ls=0.0, do_batch_norm=True, freeze=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        if self.model == 'DistMult':
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'TuckER':
            # W_torch = torch.from_numpy(np.load(w_matrix))
            # self.W = nn.Parameter(
            #     torch.Tensor(W_torch), 
            #     requires_grad = not self.freeze
            # )
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('Model is', self.model)
        self.hidden_dim = 768
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim
        
        self.num_entities = num_entities
        # self.loss = torch.nn.BCELoss(reduction='sum')
        self.loss = self.kge_loss

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        # self.pretrained_embeddings = pretrained_embeddings
        # random.shuffle(pretrained_embeddings)
        # print(pretrained_embeddings[0])
        print('Frozen:', self.freeze)
        self.node_embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_node_embeddings, dim=0), freeze=self.freeze)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        print(self.node_embedding.weight.shape)
        self.relation_embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_relation_embeddings, dim=0), freeze=False)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # self.embedding.weight.requires_grad = False
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512

        # self.lin1 = nn.Linear(self.hidden_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        # self.lin4 = nn.Linear(self.mid3, self.mid4)
        # self.hidden2rel = nn.Linear(self.mid4, self.relation_dim)
        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        self.hidden2rel_base = nn.Linear(self.mid2, self.relation_dim)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.node_embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.node_embedding.weight.size(1))
            self.bn00 = torch.nn.BatchNorm1d(self.node_embedding.weight.size(1))
            self.bn22 = torch.nn.BatchNorm1d(self.node_embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)
            self.bn00 = torch.nn.BatchNorm1d(multiplier)
            self.bn22 = torch.nn.BatchNorm1d(multiplier)



        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)        
        self._klloss = torch.nn.KLDivLoss(reduction='sum')


        # transformer goes here
        self.transformer = TransformerEncoder(self.node_embedding.weight.size(1), self.node_embedding.weight.size(1), 0.01, 0.01, 16, 4)
        self.transformer_predictor = ThinMulticlassClassification(self.node_embedding.weight.size(1))

        self.position_embedding = torch.Tensor([[[[0], [1], [2]]]]).to(self.device)

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()
        self.bn00.eval()
        self.bn22.eval()

    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.lin3(outputs)
        # outputs = F.relu(outputs)
        # outputs = self.lin4(outputs)
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def TuckER(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        x = head.view(-1, 1, head.size(1))

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, head.size(1), head.size(1))
        W_mat = self.rel_dropout(W_mat)
        x = torch.bmm(x, W_mat) 
        x = x.view(-1, head.size(1)) 
        x = self.bn2(x)
        x = self.score_dropout(x)

        x = torch.mm(x, self.node_embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def RESCAL(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        ent_dim = head.size(1)
        head = head.view(-1, 1, ent_dim)
        relation = relation.view(-1, ent_dim, ent_dim)
        relation = self.rel_dropout(relation)
        x = torch.bmm(head, relation) 
        x = x.view(-1, ent_dim)  
        x = self.bn2(x)
        x = self.score_dropout(x)
        x = torch.mm(x, self.node_embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.node_embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    
    def SimplE(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = torch.mm(s, self.node_embedding.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred



    def ComplEx(self, head, relation, kgc=False, use_sigmoid=False):
        kgc = False
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            if kgc:
                head = self.bn00(head)
            else:
                head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.node_embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)

        if self.do_batch_norm:
            if kgc:
                score = self.bn22(score)
            else:
                score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        pred = score
        use_sigmoid = False
        if use_sigmoid:
            return torch.sigmoid(pred)
        else:
            return pred


    
    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding
    

    def relational_chain_reasoning(self, head_embedding, relation_embeddings, tail_embedding, question_embedding, topk):

        aa = [head_embedding]
        aa.extend(relation_embeddings)
        xx = torch.stack(aa, dim=1)
        rel_embedding = xx.unsqueeze(1).expand(-1, topk, -1, -1)
        xx = torch.cat([rel_embedding, tail_embedding.unsqueeze(2)], dim=2)
        a, b, c, d = xx.size()
        
        xx = xx + self.position_embedding
        xx = xx.view(-1, c, d)
        out = self.transformer(xx)
        pre_tail = out[:, -1]

        score = self.transformer_predictor(pre_tail)
        score = score.view(a, -1)
        return score
    
    def relational_chain_reasoning2(self, head_embedding, relation_embeddings, tail_embedding, question_embedding, topk):
        batch_size, rel_num, dim = relation_embeddings.size()
        relation_embeddings = relation_embeddings.view(-1, dim)
        head_embedding = head_embedding.unsqueeze(1).expand(-1, rel_num, -1).reshape(-1, dim)
        aa = [head_embedding]

        aa.extend([relation_embeddings])
        xx = torch.stack(aa, dim=1)
        rel_embedding = xx.unsqueeze(1).expand(-1, topk, -1, -1)

        a_, b_, c_ = tail_embedding.size()
        tail_embedding = tail_embedding.unsqueeze(1).expand(-1, rel_num, -1, -1).reshape(-1, b_, c_)

        xx = torch.cat([rel_embedding, tail_embedding.unsqueeze(2)], dim=2)
        a, b, c, d = xx.size()
        
        xx = xx + self.position_embedding
        xx = xx.view(-1, c, d)
        out = self.transformer(xx)
        pre_tail = out[:, -1]

        score = self.transformer_predictor(pre_tail)
        score = score.view(batch_size, rel_num, -1)
        score, _idx = torch.max(score, 1)
        return score
    

    def beam_search(self, head_index, head_embedidng, path_embeddings, beam_width=5, use_sigmoid=False):
        (batch_size, embed_size) = head_embedidng.size()
        
        # the logic for the first relation and others are different.
        rel_embedding = path_embeddings[0]
        _, rel_num, dim = rel_embedding.size()
        head_embedidng = head_embedidng.unsqueeze(1).expand(-1, rel_num, -1).reshape(-1, embed_size)
        rel_embedding = rel_embedding.view(-1, dim)

        score = self.getScores(head_embedidng, rel_embedding, kgc=True)
        #score = score.view(batch_size, rel_num, -1)
        
        return score
    

    def forward(self, question_tokenized, attention_mask, p_head, p_tail, p_rels, train_transformer=False):    
        if train_transformer:
            p_head = self.node_embedding(p_head)
            #path_embdding = torch.matmul(p_rels, self.relation_embedding.weight)
            path_embdding = self.relation_embedding(p_rels)
            path_embdding = torch.mean(path_embdding, 1)
            actual = p_tail
            TOPK = 5
            top_score, top_idx = torch.topk(p_tail, k=TOPK, largest=True, sorted=True)
            tail_embedding = self.node_embedding(top_idx)
            a_ = self.relational_chain_reasoning(p_head, [path_embdding], tail_embedding, None, TOPK)
            a_actual = torch.gather(actual, 1, top_idx)

            relation_reasoning_loss = self.loss(a_, a_actual)
            return relation_reasoning_loss


        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        rel_embedding = self.applyNonLinear(question_embedding)
        p_head = self.node_embedding(p_head)
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(pred, actual)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.node_embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        
        #################################################################
        TOPK = 5
        #top_score, top_idx = torch.topk(pred, k=TOPK, largest=True, sorted=True)
        #tail_embedding = self.node_embedding(top_idx)

        p_head = p_head

        path_embdding = self.relation_embedding(p_rels)
        beam_path_embedding = path_embdding
        path_embdding = torch.mean(path_embdding, 1)

        #a_ = self.relational_chain_reasoning(p_head, [path_embdding], tail_embedding, None, TOPK)
        #a_actual = torch.gather(actual, 1, top_idx)

        #relation_reasoning_loss = self.loss(a_, a_actual)


        kbc_score = self.beam_search(None, p_head, [beam_path_embedding])
        _, rel_num, dim = beam_path_embedding.size()
        actual = actual.unsqueeze(1).expand(-1, rel_num, -1).reshape(-1, self.num_entities)
        kbc_loss = self.loss(kbc_score, actual)

        return loss + kbc_loss
        

    def get_score_ranked(self, head, question_tokenized, attention_mask, p_rels):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        rel_embedding = self.applyNonLinear(question_embedding)
        head = self.node_embedding(head).unsqueeze(0)
        scores = self.getScores(head, rel_embedding, use_sigmoid=False)
        top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        #############################################################################
        TOPK = 5
        top_score, top_idx = torch.topk(scores, k=TOPK, largest=True, sorted=True)
        tail_embedding = self.node_embedding(top_idx)

        path_embdding = self.relation_embedding(p_rels)
        path_embdding = torch.mean(path_embdding, 0)
        path_embdding = path_embdding.unsqueeze(0)

        #path_embdding = torch.matmul(p_rels, self.relation_embedding.weight).unsqueeze(0)
        a_ = self.relational_chain_reasoning(head, [path_embdding], tail_embedding, None, TOPK)
        relation_reasoning_score, relation_reasoning_idx = torch.topk(a_, k=2, largest=True, sorted=True)
        a_actual = torch.gather(top_idx, 1, relation_reasoning_idx)

        return top2, (relation_reasoning_score, a_actual)

        #return scores
        
    def get_kb_test_score_ranked(self, head, rel):
        p_head = self.node_embedding(head)
        rel_embedding = self.relation_embedding(rel)
        
        score = self.getScores(p_head, rel_embedding, kgc=True)
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2