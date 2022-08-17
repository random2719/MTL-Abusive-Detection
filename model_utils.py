import torch
import torch.nn as nn
import torch.nn.functional as F



#mask_logits(dot_prod, dmask)
#如果这i,j两点之间没有边，就被赋值无穷小
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30) 

class RelationAttention(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''
    def __init__(self, in_dim = 300, mem_dim = 300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v) # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature) # (N, L, D)
        Q = self.linear(Q) # (N, L, D)
        feature = self.linear(feature) # (N, L, D)

        att_feature = torch.cat([feature, Q], dim = 2) # (N, L, 2D)
        att_weight = self.fc(att_feature) # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out


class DotprodAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = aspect_v
        Q = Q.unsqueeze(2)  # (N, D, 1)
        dot_prod = torch.bmm(feature, Q)  # (N, L, 1)feature*aspect_feature
        dmask = dmask.unsqueeze(2)  # (N, D, 1)
        attention_weight = mask_logits(dot_prod, dmask)  # (N, L ,1)
        attention = F.softmax(attention_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)  (16,27,400) （16，27，1） （16，400，1）
        out = out.squeeze(2)
        print('out_shape=',out.shape)
        # out = F.sigmoid(out)
        # (N, D), ([N, L]), (N, L, 1)
        return out

class DotprodAttention_3(nn.Module):
    """
    GAT module operated on graphs
    """
    #768 400 500（pos）
    def __init__(self, args,in_dim=768, hidden_size=64, mem_dim=768, num_layers=2):
        super().__init__()
        self.args = args
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)
        

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
        self.bn_layer=nn.BatchNorm1d(mem_dim)

    def forward(self, feature,aspect,adj):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        #print('dmask_shape=',dmask.shape)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature) # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)

            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        node_feature=feature

        graph_feature=node_feature.sum(dim=1)
        graph_feature=self.bn_layer(graph_feature)
        return graph_feature,node_feature



class DotprodAttention_node(nn.Module):
    """
    GAT module operated on graphs
    """
    #768 400 500（pos）
    def __init__(self, args,in_dim=768, hidden_size=64, mem_dim=768, num_layers=2):
        super().__init__()
        self.args = args
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
    #def forward(self, feature,adj):
    def forward(self, feature,aspect,adj):
        print('adj_shape=',adj.shape)
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature) # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        node_feature=feature
        return node_feature

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepparseMultiHeadAttention(nn.Module):
    def __init__(self, h=6, Co=300, cat=True):
        super().__init__()
        self.hidden_size = Co // h
        self.h = h
        self.fc1 = nn.Linear(Co, Co)
        self.relu = nn.ReLU()
        self.fc2s = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(h)])
        self.cat = cat

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        nbatches = dep_tags_v.size(0)
        Q = self.fc1(dep_tags_v).view(nbatches, -1, self.h,
                                      self.hidden_size)  # [N, L, #heads, hidden_size]
        Q = self.relu(Q)
        Q = Q.transpose(0, 2)  # [#heads, L, N, hidden_size]
        Q = [l(q).squeeze(2).transpose(0, 1)
             for l, q in zip(self.fc2s, Q)]  # [N, L] * #heads
        Q = [F.softmax(mask_logits(q, dmask), dim=1).unsqueeze(2)
             for q in Q]  # [N, L, 1] * #heads

        if self.cat:
            out = torch.cat(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=1)
        else:
            out = torch.stack(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=2)
            out = torch.sum(out, dim=2)

        return out, Q[0]  # ([N, L]) one head



class RelationAttention_2(nn.Module):
    """
    Relation gat model, use the embedding of the edges to predict attention weight
    """
    #bert:768
    def __init__(self, args, dep_rel_num,in_dim=400, hidden_size=64, mem_dim=400,num_layers=2):
        super(RelationAttention_2, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # 新加的
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
        self.bn_layer=nn.BatchNorm1d(mem_dim)

    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)


        # gcn layer
        for l in range(self.num_layers):
            # relation based GAT, attention over relations

            h = self.W[l](feature)  # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N * N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            rel_adj_logits=e

            dmask = adj.view(B, -1)  # (batch_size, n*n)
            rel_adj_logits = F.softmax(mask_logits(rel_adj_logits, dmask), dim=1)
            rel_adj_logits = rel_adj_logits.view(*rel_adj.size())  # (batch_size, n, n)

            Ax = rel_adj_logits.bmm(feature)
            feature=Ax
            feature = self.dropout(Ax) if l < self.num_layers - 1 else Ax


        feature = feature.sum(dim=1)
        feature=self.bn_layer(feature)

        return feature
