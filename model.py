import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits,DotprodAttention_3,RelationAttention_2,DotprodAttention_node
from tree import *


class Node_classification(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Node_classification, self).__init__()
        self.args = args
        #节点分类
        self.dropout = nn.Dropout(args.dropout)
        last_hidden_size_node = args.hidden_size * 2

        layers_node = [
            nn.Linear(last_hidden_size_node, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers_node += [nn.Linear(args.final_hidden_size,
                                args.final_hidden_size), nn.ReLU()]
        self.fcs_node = nn.Sequential(*layers_node)
        self.fc_final_node = nn.Linear(args.final_hidden_size, args.num_classes)
    def forward(self,node_feature_final):
        #节点分类
        node_x = self.dropout(node_feature_final)
        node_x = self.fcs_node(node_x)
        node_logit = self.fc_final_node(node_x)
        print('logit_node_shape=',node_logit.shape)
        return node_logit

    

class GAT_ours_node(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(GAT_ours_node, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        #self.gat_dep = [RelationAttention_2(args,dep_tag_num).to(args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention_node(args).to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        #self.dep_embed = nn.Embedding(dep_tag_num, args.dep_relation_embed_dim)

        last_hidden_size = args.hidden_size * 2

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        #fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        fmask, rel_adj = inputs_to_deprel_adj(dep_heads, dep_rels, text_len, )
        #print('fmask_shape=',fmask.shape)
        #print('fmask=',fmask)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fmask=fmask.float().to(device)
        rel_adj=rel_adj.float().to(device)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            #aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        #print('feature_shape=',feature.shape)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature,aspect_feature,fmask) for g in self.gat]
            for a in range(gat_out[0].shape[0]):#16
                for i in range(len(gat_out)):#6
                    sum=torch.zeros(gat_out[i].shape[1],gat_out[i].shape[2]).to('cuda')
                    for b in range(gat_out[i].shape[1]):#(15,400)
                        sum=sum+gat_out[i][a][b]
                sum=sum/6
                gat_out[0][a]=sum
            node_feature=gat_out[0]
            
        x = self.dropout(node_feature)
        x = self.fcs(x)
        logit = self.fc_final(x)
        print('logit_shape=',logit.shape)
        return logit



class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        layers = [nn.Linear(
            config.hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, args.num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_cat_ids, segment_ids):
        #outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        pooled_output = outputs[1]
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits



def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class Node_classification_2(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Node_classification_2, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = nn.ModuleList([RelationAttention_2(args,dep_tag_num).to(args.device) for i in range(args.num_heads)])
        if args.gat_attention_type == 'linear':
            self.gat = nn.ModuleList([LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] )# we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = nn.ModuleList([DotprodAttention_3(args).to(args.device) for i in range(args.num_heads)])
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.dep_relation_embed_dim)

        #节点分类
        #########################
        last_hidden_size_node = args.hidden_size * 2

        layers_node = [
            nn.Linear(last_hidden_size_node, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers_node += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs_node = nn.Sequential(*layers_node)
        self.fc_final_node = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs,important_word_ids):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        #fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        fmask, rel_adj = inputs_to_deprel_adj(dep_heads, dep_rels, text_len, )
        #print('fmask_shape=',fmask.shape)
        #print('fmask=',fmask)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fmask=fmask.float().to(device)
        rel_adj=rel_adj.float().to(device)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect = aspect.to(torch.int64)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            #aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        #print('feature_shape=',feature.shape)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        #dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = [g(fmask, rel_adj, dep_feature).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        #dep_out = [g(fmask, rel_adj, dep_feature) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) # (N, H, D)
        dep_out = dep_out.mean(dim = 1) # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out=[]
            node_feature_list=[]
            #print('gat=',self.gat)
            
            for g in self.gat:
                graph_feature,node_feature=g(feature,aspect_feature,fmask)
                gat_out.append(graph_feature.unsqueeze(1))
                node_feature_list.append(node_feature)

            #gat_out = [g(feature,aspect_feature,fmask) for g in self.gat]
            for a in range(node_feature_list[0].shape[0]):#16
                for i in range(len(node_feature_list)):#6
                    sum=torch.zeros(node_feature_list[i].shape[1],node_feature_list[i].shape[2]).to('cuda')
                    for b in range(node_feature_list[i].shape[1]):#(15,400)
                        sum=sum+node_feature_list[i][a][b]
                sum=sum/6
                node_feature_list[0][a]=sum
            node_feature_final=node_feature_list[0]

            
                

        #节点分类
        node_x = self.dropout(node_feature_final)
        node_x = self.fcs_node(node_x)
        node_logit = self.fc_final_node(node_x)
        logit=0




        return logit,node_logit,node_feature_final