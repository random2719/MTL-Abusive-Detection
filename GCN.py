import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules import LSTM, MLP

from tree_GCN import *

import argparse
import numpy as np
import torch





class GCNClassifier(nn.Module):
    def __init__(self, args, dep_tag_num,pos_tag_num):
        super().__init__()
        self.args = args

        in_dim = args.hidden_dim
        #in_dim = args.hidden_s*2
        self.gcn_model = GCNAbsaModel(args, dep_tag_num,pos_tag_num)
        self.classifier = MLP([in_dim, args.num_classes])

    # def forward(self, aspmask, words, aspect, seq_len, dephead, pos, post, deprel):
    #     output = self.gcn_model(aspmask, words, aspect, seq_len, dephead, pos,
    #                             post, deprel)["pred"]
    #     logit = self.classifier(output)
    #     return {"pred": logit}
    def forward(self, words, seq_len, dephead, pos, deprel):
        output = self.gcn_model(words, seq_len, dephead, pos,
                                deprel)["pred"]
        logits = self.classifier(output)
        return logits


class GCNAbsaModel(nn.Module):
    def __init__(self, args, dep_tag_num,pos_tag_num):
        super().__init__()
        self.args = args

        #self.emb_matrix = emb_matrix

        num_embeddings,embed_dim=args.glove_embedding.shape
        self.tok_emb = nn.Embedding(num_embeddings, embed_dim, padding_idx=0) #word_len,300
        self.tok_emb.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)
        # if emb_matrix is not None:
        #     self.tok_emb = emb_matrix
        self.pos_emb = (
            nn.Embedding(pos_tag_num, args.pos_dim, padding_idx=0)
            if args.pos_dim > 0
            else None
        )

        embedding = (self.tok_emb, self.pos_emb)

        # gcn
        self.gcn = GCN(args, embedding, args.hidden_dim)

    def forward(self, words, seq_len, dephead, pos, deprel):
        tok = words
        dephead = dephead
        seq_len = seq_len
        maxlen = max(seq_len)

        def inputs_to_tree_reps(dephead, words, l):
            trees = [head_to_tree(dephead[i], words[i], l[i])
                     for i in range(len(l))]
            adj = [
                tree_to_adj(
                    maxlen, tree, directed=self.args.direct, self_loop=self.args.loop
                ).reshape(1, maxlen, maxlen)
                for tree in trees
            ]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj

        adj = inputs_to_tree_reps(dephead, tok, seq_len)
        h = self.gcn(
            adj, words, seq_len, dephead, pos, deprel
        )

        output = (h["pred"]).sum(dim=1)

        return {"pred": output}


class GCN(nn.Module):
    def __init__(self, args, embedding, hid_dim):
        super(GCN, self).__init__()

        self.args = args
        self.layers = args.num_layers
        self.mem_dim = hid_dim
        #self.in_dim = args.tok_dim + args.pos_dim + args.post_dim
        self.in_dim = args.tok_dim + args.pos_dim
        #self.tok_emb, self.pos_emb, self.post_emb = embedding
        self.tok_emb, self.pos_emb = embedding
        # drop out
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # lstm
        input_size = self.in_dim
        print('input_size=',input_size)
        self.rnn = LSTM(
            input_size,
            hidden_size=args.rnn_hidden,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=True,
        )
        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # gcn layer
        self.G = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = [self.in_dim, self.mem_dim][layer != 0]
            self.G.append(MLP([input_dim, self.mem_dim]))

    def forward(self, adj, words, seq_len, dephead, pos, deprel):
        tok = words
        pos = pos

        seq_len = seq_len

        word_embs = self.tok_emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]

        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)  #input_dropout

        #print('embs=',embs.size())

        rnn_output, _ = self.rnn(embs)
        gcn_input = self.rnn_drop(rnn_output)

        #print('gcn_input=',gcn_input.size())
        adj = adj.to(gcn_input)
        #print('adj=',adj.size())

        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1
        for l in range(self.layers):
            AX = adj.bmm(gcn_input)
            AXW = self.G[l](AX)
            AXW = F.relu(AXW / denom)
            gcn_input = self.gcn_drop(AXW) if l < self.layers - 1 else AXW
        return {"pred": gcn_input}

#model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])


# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     # Required parameters
#     parser.add_argument('--dataset_name', type=str, default='hatespeech',
#                         choices=['rest', 'laptop', 'twitter', 'hatespeech'],
#                         help='Choose absa dataset.')
#     parser.add_argument('--output_dir', type=str, default='data/output-hatespeech',
#                         help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
#     parser.add_argument('--num_classes', type=int, default=3,
#                         help='Number of classes of ABSA.')
#
#     parser.add_argument('--cuda_id', type=str, default='3',
#                         help='Choose which GPUs to run')
#     parser.add_argument('--seed', type=int, default=2019,
#                         help='random seed for initialization')
#
#     # Model parameters
#     parser.add_argument('--glove_dir', type=str, default='data/glove',
#                         help='Directory storing glove embeddings')
#     parser.add_argument('--bert_model_dir', type=str, default='data/bert/bert-base-uncased',
#                         help='Path to pre-trained Bert model.')
#     parser.add_argument('--pure_bert', action='store_true',
#                         help='Cat text and aspect, [cls] to predict.')
#     parser.add_argument('--gat_bert', action='store_true',
#                         help='Cat text and aspect, [cls] to predict.')
#
#     parser.add_argument('--highway', action='store_true',
#                         help='Use highway embed.')
#
#     parser.add_argument('--num_layers', type=int, default=2,
#                         help='Number of layers of bilstm or highway or elmo.')
#
#     parser.add_argument('--add_non_connect', type=bool, default=True,
#                         help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
#     parser.add_argument('--multi_hop', type=bool, default=True,
#                         help='Multi hop non connection.')
#     parser.add_argument('--max_hop', type=int, default=4,
#                         help='max number of hops')
#
#     parser.add_argument('--num_heads', type=int, default=6,
#                         help='Number of heads for gat.')
#
#     parser.add_argument('--dropout', type=float, default=0,
#                         help='Dropout rate for embedding.')
#
#     parser.add_argument('--num_gcn_layers', type=int, default=1,
#                         help='Number of GCN layers.')
#     parser.add_argument('--gcn_mem_dim', type=int, default=300,
#                         help='Dimension of the W in GCN.')
#     parser.add_argument('--gcn_dropout', type=float, default=0.2,
#                         help='Dropout rate for GCN.')
#     # GAT
#     parser.add_argument('--gat', action='store_true',
#                         help='GAT')
#     parser.add_argument('--gat_our', action='store_true',
#                         help='GAT_our')
#     parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
#                         help='The attention used for gat')
#
#     parser.add_argument('--embedding_type', type=str, default='glove', choices=['glove', 'bert'])
#     parser.add_argument('--embedding_dim', type=int, default=300,
#                         help='Dimension of glove embeddings')
#     parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
#                         help='Dimension for dependency relation embeddings.')
#
#     parser.add_argument('--hidden_size', type=int, default=300,
#                         help='Hidden size of bilstm, in early stage.')
#     parser.add_argument('--final_hidden_size', type=int, default=300,
#                         help='Hidden size of bilstm, in early stage.')
#     parser.add_argument('--num_mlps', type=int, default=2,
#                         help='Number of mlps in the last of model.')
#     # 新加的
#     parser.add_argument("--hidden_dim", type=int, default=50, help="GCN mem dim.")
#     parser.add_argument("--pos_dim",
#                         type=int,
#                         default=30,
#                         help="Pos embedding dimension.")
#
#     # Training parameters
#     parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
#                         help="Batch size per GPU/CPU for training.")
#     parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
#                         help="Batch size per GPU/CPU for evaluation.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", default=1e-3, type=float,
#                         help="The initial learning rate for Adam.")
#
#     parser.add_argument("--weight_decay", default=0.0, type=float,
#                         help="Weight deay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
#     parser.add_argument("--num_train_epochs", default=30.0, type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
#     parser.add_argument('--logging_steps', type=int, default=50,
#                         help="Log every X updates steps.")
#
#     return parser.parse_args()


# args = parse_args()
# args.device = 'cpu'
# train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(args)
# train_sampler = RandomSampler(train_dataset)
# collate_fn = get_collate_fn(args)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
#                               batch_size=16,
#                               collate_fn=collate_fn)
# train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
# model=GCNClassifier()
# for _ in train_iterator:
#     # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
#     for step, batch in enumerate(train_dataloader):
#         batch = tuple(t.to(args.device) for t in batch)
#         inputs, labels = get_input_from_batch(args, batch)
#         adj, rel_adj = inputs_to_deprel_adj(args, inputs['dep_heads'], inputs['dep_rels'], inputs['text_len'])
#         head_to_tree(inputs['dep_heads'],)
#         with open('write.txt','w',encoding='utf-8') as f:
#             f.write(str(adj))
#             f.write('\n')
#             f.write(str(rel_adj))
#             f.write('\n')