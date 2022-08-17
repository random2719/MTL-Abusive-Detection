import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter

#from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits,DotprodAttention_3,RelationAttention_2,DotprodAttention_node
from tree import *

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

class Expert_graph(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Expert_graph, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = nn.ModuleList([RelationAttention_2(args,dep_tag_num).to(args.device) for i in range(args.num_heads)])
        if args.gat_attention_type == 'linear':
            self.gat = nn.ModuleList([LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)]) # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = nn.ModuleList([DotprodAttention_3(args).to(args.device) for i in range(args.num_heads)])
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.dep_relation_embed_dim)


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
        #print('dep_heads=',dep_heads)
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

        
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        #图分类
        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')  [16,800]
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        node_x = self.dropout(node_feature_final)
        return x

class Expert_node(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Expert_node, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = nn.ModuleList([RelationAttention_2(args,dep_tag_num).to(args.device) for i in range(args.num_heads)])
        if args.gat_attention_type == 'linear':
            self.gat = nn.ModuleList([LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)]) # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = nn.ModuleList([DotprodAttention_3(args).to(args.device) for i in range(args.num_heads)])
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.dep_relation_embed_dim)


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

        fmask, rel_adj = inputs_to_deprel_adj(dep_heads, dep_rels, text_len, )

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


        feature, _ = self.bilstm(feature) # (N,L,D)
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

        
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        #图分类
        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')  [16,800]
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        node_x = self.dropout(node_feature_final)
        return x,node_x



class Tower_graph(nn.Module):
    def __init__(self, args):
        super(Tower_graph, self).__init__()
        self.args = args
        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
    def forward(self, x):
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class Tower_node(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args):
        super(Tower_node, self).__init__()
        self.args = args
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

                
    def forward(self, node_feature_final):
        #节点分类
        node_feature_final = self.fcs_node(node_feature_final)
        node_logit = self.fc_final_node(node_feature_final)
        return node_logit

class MMOE(nn.Module):
    def __init__(self, args, num_experts, dep_tag_num, pos_tag_num, towers_graph_hidden, towers_node_hidden,tasks):
        super(MMOE, self).__init__()
        self.args = args
        self.num_experts = num_experts
        self.dep_tag_num = dep_tag_num
        self.pos_tag_num = pos_tag_num  #
        self.towers_graph_hidden = towers_graph_hidden
        self.towers_node_hidden = towers_node_hidden
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert(self.args, self.dep_tag_num, self.pos_tag_num) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(args.embedding_dim, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers_graph = Tower_graph(self.args)
        self.towers_node = Tower_node(self.args)


    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        experts_o = [e(sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        gates_o=[]
        gates_o = [self.softmax(g(sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs)) for g in self.w_gates]

        tower_input_graph = [g.t().unsqueeze(2).expand(-1, -1, self.towers_graph_hidden) * experts_o_tensor for g in gates_o]
        tower_input_graph = [torch.sum(ti, dim=0) for ti in tower_input_graph]

        tower_input_node = [g.t().unsqueeze(2).expand(-1, -1, self.towers_node_hidden) * experts_o_tensor for g in gates_o]
        tower_input_node = [torch.sum(ti, dim=0) for ti in tower_input_graph]

        final_output_graph = [t(ti) for t, ti in zip(self.towers_graph, tower_input_graph)]
        final_output_node = [t(ti) for t, ti in zip(self.towers_node, tower_input_node)]


        return final_output_graph,final_output_node


class GatingMechanism(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(GatingMechanism, self).__init__()
        self.args = args
        self.dep_tag_num=dep_tag_num
        self.pos_tag_num=pos_tag_num

        # gating 的参数
        self.gate_theta_1 = Parameter(torch.empty(1, 800))
        self.gate_theta_2 = Parameter(torch.empty(1, 800))

        nn.init.xavier_uniform_(self.gate_theta_1)
        nn.init.xavier_uniform_(self.gate_theta_2)
        #share_layer
        self.expert_graph=Expert_graph(self.args, self.dep_tag_num, self.pos_tag_num)
        self.expert_node=Expert_node(self.args, self.dep_tag_num, self.pos_tag_num)

        #分开的层
        self.towers_graph = Tower_graph(self.args)
        self.towers_node = Tower_node(self.args)


    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        graph_X=self.expert_graph(sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs)
        node_Y,node_no_mean=self.expert_node(sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs)
        graph_X=graph_X.unsqueeze(1)
        gate_1=self.gate_theta_1
        node_Y_mean=node_Y.unsqueeze(1)
        gate_2=self.gate_theta_2

        output_graph = torch.mul(gate_1, graph_X) + torch.mul(gate_2, node_Y_mean)
        output_graph=output_graph.squeeze(1)
        logit=self.towers_graph(output_graph)
        node_logit=self.towers_node(node_no_mean)


        return logit,node_logit