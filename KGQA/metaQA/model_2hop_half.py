import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
import random
from transformer import TransformerEncoder

class MulticlassClassification(nn.Module):
    def __init__(self, input_dim, num_class):
        super(MulticlassClassification, self).__init__()

        #self.layer_1 = nn.Linear(num_feature, num_feature)
        self.layer_2 = nn.Linear(input_dim, int(input_dim * 1.2))
        self.layer_3 = nn.Linear(int(input_dim * 1.2), int(num_class * 0.2))
        self.layer_out = nn.Linear(int(num_class * 0.2), num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.batchnorm2 = nn.BatchNorm1d(int(input_dim * 1.2))
        self.batchnorm3 = nn.BatchNorm1d(int(num_class * 0.2))
        self.m = nn.Softmax()

    def forward(self, x):
        #x = self.layer_1(x)
        #x = self.batchnorm1(x)
        #x = self.relu(x)
        x = x.squeeze(1)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        x = self.m(x)

        x = x.unsqueeze(1)

        return x


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
        x = self.m(x)

        return x


class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, relation_dim, num_entities, pretrained_node_embeddings, pretrained_relation_embeddings, 
        device, entdrop, reldrop, scoredrop, l3_reg, model, ls, w_matrix, bn_list, freeze=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.bn_list = bn_list
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        if self.model == 'DistMult':
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'Rotat3':
            multiplier = 3
            self.getScores = self.Rotat3
        elif self.model == 'TuckER':
            W_torch = torch.from_numpy(np.load(w_matrix))
            self.W = nn.Parameter(
                torch.Tensor(W_torch), 
                requires_grad = True
            )
            # self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)), 
            #                         dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('Model is', self.model)
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = 1
        self.bidirectional = True
        
        self.num_entities = num_entities
        self.loss = torch.nn.BCELoss(reduction='sum')

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        print('Frozen:', self.freeze)
        self.pretrained_node_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_node_embeddings), freeze=self.freeze)
        self.pretrained_relation_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_relation_embeddings), freeze=False)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 256
        self.mid2 = 256

        self.lin1 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)
        self.lin2 = nn.Linear(self.mid1, self.mid2, bias=False)
        xavier_normal_(self.lin1.weight.data)
        xavier_normal_(self.lin2.weight.data)
        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)
        self.hidden2rel_base = nn.Linear(hidden_dim * 2, self.relation_dim)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.pretrained_node_embeddings.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.pretrained_node_embeddings.weight.size(1))
            self.bn00 = torch.nn.BatchNorm1d(self.pretrained_node_embeddings.weight.size(1))
            self.bn22 = torch.nn.BatchNorm1d(self.pretrained_node_embeddings.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)
            self.bn00 = torch.nn.BatchNorm1d(multiplier)
            self.bn22 = torch.nn.BatchNorm1d(multiplier)

        for i in range(3):
            for key, value in self.bn_list[i].items():
                self.bn_list[i][key] = torch.Tensor(value).to(device)

        
        self.bn0.weight.data = self.bn_list[0]['weight'].clone()
        self.bn0.bias.data = self.bn_list[0]['bias'].clone()
        self.bn0.running_mean.data = self.bn_list[0]['running_mean'].clone()
        self.bn0.running_var.data = self.bn_list[0]['running_var'].clone()

        self.bn2.weight.data = self.bn_list[2]['weight'].clone()
        self.bn2.bias.data = self.bn_list[2]['bias'].clone()
        self.bn2.running_mean.data = self.bn_list[2]['running_mean'].clone()
        self.bn2.running_var.data = self.bn_list[2]['running_var'].clone()


        self.bn00.weight.data = self.bn_list[0]['weight'].clone()
        self.bn00.bias.data = self.bn_list[0]['bias'].clone()
        self.bn00.running_mean.data = self.bn_list[0]['running_mean'].clone()
        self.bn00.running_var.data = self.bn_list[0]['running_var'].clone()

        self.bn22.weight.data = self.bn_list[2]['weight'].clone()
        self.bn22.bias.data = self.bn_list[2]['bias'].clone()
        self.bn22.running_mean.data = self.bn_list[2]['running_mean'].clone()
        self.bn22.running_var.data = self.bn_list[2]['running_var'].clone()


        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)

        # kgc
        self.GRU_decoder = nn.LSTM(self.relation_dim, self.relation_dim, self.n_layers, bidirectional=False, batch_first=True)
        self.lin1_e2d = nn.Linear(self.hidden_dim, self.mid1, bias=False)
        self.lin2_e2d = nn.Linear(self.mid1, self.mid2, bias=False)
        xavier_normal_(self.lin1_e2d.weight.data)
        xavier_normal_(self.lin2_e2d.weight.data)
        self.hidden2hidden = nn.Linear(self.mid2, self.relation_dim)
        #self.fc_out = nn.Linear(self.relation_dim, self.pretrained_relation_embeddings.weight.shape[0])
        self.path_decoder_fc_out = MulticlassClassification(self.relation_dim, self.pretrained_relation_embeddings.weight.shape[0])

        self.decoder_SOS = nn.Embedding(1, self.pretrained_relation_embeddings.weight.shape[1])
        nn.init.xavier_uniform(self.decoder_SOS.weight)

        #self.score_predictor = ThinMulticlassClassification(self.num_entities * 2, self.num_entities)

        self.transformer = TransformerEncoder(self.relation_dim, self.relation_dim, 0.01, 0.01, 16, 4)
        self.transformer_predictor = ThinMulticlassClassification(self.relation_dim)

        self.relation_reasoning_gru = nn.LSTM(self.relation_dim, self.relation_dim, self.n_layers, bidirectional=False, batch_first=True)
        self.relation_reasoning_predictor = ThinMulticlassClassification(self.relation_dim)

        self.position_embedding = torch.Tensor([[[[0], [1], [2], [3]]]]).to(self.device)
        

    def applyNonLinear(self, outputs):
        outputs = self.lin1(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin2(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs
    
    def applyNonLinear_e2d(self, outputs):
        outputs = self.lin1_e2d(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin2_e2d(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2hidden(outputs)
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

        x = torch.mm(x, self.pretrained_node_embeddings.weight.transpose(1,0))
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
        x = torch.mm(x, self.pretrained_node_embeddings.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.pretrained_node_embeddings.weight.transpose(1,0))
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
        s = torch.mm(s, self.pretrained_node_embeddings.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx(self, head, relation, kgc=False, use_sigmoid=True):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
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

        if kgc:
            with torch.no_grad():
                re_tail, im_tail = torch.chunk(self.pretrained_node_embeddings.weight, 2, dim =1)
        else:
            re_tail, im_tail = torch.chunk(self.pretrained_node_embeddings.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)

        if kgc:
            score = self.bn22(score)
        else:
            score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        if use_sigmoid:
            pred = torch.sigmoid(score)
            return pred
        else:
            return score

    def Rotat3(self, head, relation):
        pi = 3.14159265358979323846
        relation = F.hardtanh(relation) * pi
        r = torch.stack(list(torch.chunk(relation, 3, dim=1)), dim=1)
        h = torch.stack(list(torch.chunk(head, 3, dim=1)), dim=1)
        h = self.bn0(h)
        h = self.ent_dropout(h)
        r = self.rel_dropout(r)
        
        r = r.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        x = h[0]
        y = h[1]
        z = h[2]

        # need to rotate h by r
        # r contains values in radians

        for i in range(len(r)):
            sin_r = torch.sin(r[i])
            cos_r = torch.cos(r[i])
            if i == 0:
                x_n = x.clone()
                y_n = y * cos_r - z * sin_r
                z_n = y * sin_r + z * cos_r
            elif i == 1:
                x_n = x * cos_r - y * sin_r
                y_n = x * sin_r + y * cos_r
                z_n = z.clone()
            elif i == 2:
                x_n = z * sin_r + x * cos_r
                y_n = y.clone()
                z_n = z * cos_r - x * sin_r

            x = x_n
            y = y_n
            z = z_n

        s = torch.stack([x, y, z], dim=1)        
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = s.permute(1, 0, 2)
        s = torch.cat([s[0], s[1], s[2]], dim = 1)
        ans = torch.mm(s, self.pretrained_node_embeddings.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    

    def attention_path_decoder(self, hidden, cell_state, batch_size, hop_num, SOS=None, path=None, teacher_forcing_ratio=0.5):
        # KBC here
        # decode question embedding
        # bidirected, so it is two hidden cat
        # dimension is batch_size x 2 * hidden_dim
        path_prediction_list = []
        decoder_hidden = self.applyNonLinear_e2d(hidden[0,:,:] + hidden[1,:,:]).unsqueeze(0)
        decoder_celll = self.applyNonLinear_e2d(cell_state[0,:,:] + cell_state[1,:,:]).unsqueeze(0)
        if SOS is None:
            decoder_input_idx = torch.zeros(1, batch_size, dtype=torch.long).to(self.device)
            decoder_input = self.decoder_SOS(decoder_input_idx).squeeze(0)
        else:
            decoder_input = SOS
        STEP = hop_num
        predicted_relations = []
        for t in range(STEP):
            decoder_input = decoder_input.unsqueeze(1)
            output, (hidden, cell) = self.GRU_decoder(decoder_input, (decoder_hidden, decoder_celll))
            # predict according to output
            prediction = self.path_decoder_fc_out(output).squeeze(1)
            path_prediction_list.append(prediction)
            # prediction to next input
            if path is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = prediction.argmax(1)
                decoder_input_idx = path[:,t] if teacher_force else top1
            else:
                decoder_input_idx = prediction.argmax(1)
            decoder_input = self.pretrained_relation_embeddings(decoder_input_idx)
            
            predicted_relations.append(decoder_input)
            decoder_hidden = hidden
            decoder_celll = cell
        return predicted_relations, path_prediction_list
    
    def beam_search(self, head_index, head_embedidng, path_embeddings, beam_width=50, use_sigmoid=True):
        (batch_size, embed_size) = head_embedidng.size()
        
        # the logic for the first relation and others are different.
        rel_embedding = path_embeddings[0]
        
        if len(path_embeddings) == 1:
            if use_sigmoid:
                score = self.getScores(head_embedidng, rel_embedding, kgc=True)
            else:
                score = self.getScores(head_embedidng, rel_embedding, use_sigmoid=False, kgc=True)
            return score
        else:
            score = self.getScores(head_embedidng, rel_embedding, kgc=True)
        
        head_index = head_index.unsqueeze(1)
        for idx in range(1, len(path_embeddings)):
            # get head emebddings
            # here can use .scatter_ function
            # delete self
            score.scatter_(1, head_index, 0)

            #for i in range(len(head_index)):
            #    score[i, head_index[i]] = 0

            # select top k, k=beam_width
            # k should not include head itself
            topk = torch.topk(score, k=beam_width, largest=True, sorted=True)
            head_index = topk[1]

            rel_embedding = path_embeddings[idx]
            # expand rel_embedding, copy it multi times? or not
            # rel_embedding: batch_size x beam_width x embed_size
            rel_embedding = rel_embedding.unsqueeze(1).repeat((1, beam_width, 1))
            
            # new_head_emb: batch_size x beam_width x embed_size
            with torch.no_grad():
                new_head_emb = self.pretrained_node_embeddings(head_index)
            # expand new_head_emb and rel_embedding
            head_embs = new_head_emb.view(-1, embed_size)
            relation_embs = rel_embedding.view(-1, embed_size)

            if idx == len(path_embeddings)-1 and not use_sigmoid:
                scores = self.getScores(head_embs, relation_embs, kgc=True, use_sigmoid=False)
            else:
                scores = self.getScores(head_embs, relation_embs, kgc=True)
            scores = scores.view(batch_size, -1, self.num_entities)
            # find max: d: batch_size x num_entities
            (d, e) = torch.max(scores, 1)
            score = d

        return score
    

    def refinement(self, head_embedding, relation_embeddings, tail_embedding, question_embedding, topk):

        # aa = [head_embedding]
        # aa.extend(relation_embeddings)
        # xx = torch.stack(aa, dim=1)
        # rel_embedding = xx.unsqueeze(1).expand(-1, topk, -1, -1)
        # xx = torch.cat([rel_embedding, tail_embedding.unsqueeze(2)], dim=2)
        # a, b, c, d = xx.size()
        
        # xx = xx + self.position_embedding
        # xx = xx.view(-1, c, d)
        # out = self.transformer(xx)
        # pre_tail = out[:, -1]

        # score = self.transformer_predictor(pre_tail)
        # score = score.view(a, -1)
        # return score

        ######
        batch_size = head_embedding.size()[0]
        decoder_hidden = torch.zeros_like(head_embedding, device=self.device).unsqueeze(0)
        decoder_celll = torch.zeros_like(head_embedding, device=self.device).unsqueeze(0)
        for e in relation_embeddings:
            e = e.unsqueeze(1)
            output, (hidden, cell) = self.relation_reasoning_gru(e, (decoder_hidden, decoder_celll))
            decoder_hidden = hidden
            decoder_celll = cell

        output = output.squeeze(1)
        ######
        aa = [question_embedding]
        aa.extend([head_embedding])
        #aa.extend([question_embedding])
        aa.extend([output])

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



    def forward(self, sentence, p_head, p_tail, question_len, p_path):
        # use EmbedKGQA find answers, then use refinement to rerank
        head_index = p_head

        embeds = self.word_embeddings(sentence)
        packed_output = pack_padded_sequence(embeds, question_len.cpu(), batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # outputs = self.drop1(outputs)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)
        p_head = self.pretrained_node_embeddings(p_head)
        kgqa_score = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        
        loss = self.loss(kgqa_score, actual)

        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.pretrained_node_embeddings.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        
        kgc=True
        if kgc:
            hidden_ = hidden.clone().detach().requires_grad_(False)
            cell_state_ = cell_state.clone().detach().requires_grad_(False)

            pathes_embeddings, path_prediction_list = self.attention_path_decoder(hidden_, cell_state_, p_head.size()[0], hop_num=p_path.size()[1], SOS=None, path=p_path)

            p_head = p_head.clone().detach().requires_grad_(False)
            p_head2 = p_head.clone().detach().requires_grad_(False)
            kbc_score = self.beam_search(head_index, p_head, pathes_embeddings, use_sigmoid=True)
            kbc_loss = self.loss(kbc_score, actual)

            total_path_loss = 0
            for i in range(p_path.shape[1]):
                path_pred = path_prediction_list[i]
                # path_loss
                tmp_path = p_path[:,i].view(-1)
                path_ground_truth = F.one_hot(tmp_path, num_classes=self.pretrained_relation_embeddings.weight.shape[0])
                path_loss = self.loss(path_pred, path_ground_truth.float())
                total_path_loss = total_path_loss + path_loss

            # select top k from kgqa predict
            TOPK = 5
            top_score, top_idx = torch.topk(kgqa_score, k=TOPK, largest=True, sorted=True)
            tail_embedding = self.pretrained_node_embeddings(top_idx)

            p_head = p_head2

            a_ = self.refinement(p_head, pathes_embeddings, tail_embedding, rel_embedding, TOPK)
            a_actual = torch.gather(actual, 1, top_idx)

            relation_reasoning_loss = self.loss(a_, a_actual)

            
        
        # concat
        #all_score = torch.cat((kgqa_score, kbc_score), 1)
        #score = self.score_predictor(all_score)
        #p_loss = self.loss(kbc_score, actual)
        return loss + relation_reasoning_loss + 0.01 * kbc_loss + total_path_loss
        
    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)
        return rel_embedding

    def get_score_ranked(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)

        head = self.pretrained_node_embeddings(head).unsqueeze(0)
        score = self.getScores(head, rel_embedding)
        
        # use embedKGQA find answers
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        #return top2, (top2[0], top2[1])

        #########################################################
        hidden_ = rel_embedding.clone().detach().requires_grad_(False)
        cell_state_ = rel_embedding.clone().detach().requires_grad_(False)
        # answer is find by EmbedKGQA, but were reranked 
        pathes_embeddings, path_prediction_list = self.attention_path_decoder(hidden, cell_state, head.size()[0], hop_num=2, SOS=None)
        TOPK = 5
        top_score, top_idx = torch.topk(score, k=TOPK, largest=True, sorted=True)
        tail_embedding = self.pretrained_node_embeddings(top_idx)
        # use tranformer reranking
        a_ = self.refinement(head, pathes_embeddings, tail_embedding, rel_embedding, TOPK)
        relation_reasoning_score, relation_reasoning_idx = torch.topk(a_, k=2, largest=True, sorted=True)
        a_actual = torch.gather(top_idx, 1, relation_reasoning_idx)
        return top2, (relation_reasoning_score, a_actual)
    
    def get_kb_test_score_ranked(self, head, rel):
        p_head = self.pretrained_node_embeddings(head)
        rel_embedding = self.pretrained_relation_embeddings(rel)
        score = self.getScores(p_head, rel_embedding, kgc=True)
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2