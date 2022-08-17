import logging
import os
import random
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer
from collections import Counter
from thop import profile
#import nni
from pytorch_model_summary import summary

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10]
                    }
        labels = batch[6]
        node_labels=batch[11]
        #important_word_ids=batch[12]
    else: # bert
        if args.pure_bert:
            # input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            #input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids,node_labels
            inputs = {  'input_ids': batch[0],
                        'token_type_ids': batch[1]}
            #inputs = {  'input_cat_ids': batch[4],
                        #'token_type_ids': batch[5]}
            labels = batch[6]
            node_labels=batch[11]
        else:
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            #input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids,  dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids,node_labels
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_cat_ids': batch[4],
                        'segment_ids': batch[5],
                        'dep_tags': batch[6],
                        'pos_class': batch[7],
                        'text_len': batch[8],
                        'aspect_len': batch[9],
                        'dep_rels': batch[11],
                        'dep_heads': batch[12],
                        'aspect_position': batch[13],
                        'dep_dirs': batch[14]}
            labels = batch[10]
            node_labels=batch[15]
    return inputs, labels,node_labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


def train_multi_task(args, train_dataset, model, dev_dataset,test_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()
    #print('*'*20, "模型架构:", '*'*20, '\n', model, '\n')
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        # lambd = torch.tensor(1.)
        # l2_reg = torch.tensor(0.)
        # for param in model.parameters():
        #     l2_reg += torch.norm(param)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    eval_best_loss = float('inf')
    eval_f1 = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    all_test_results=[]
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels,node_labels = get_input_from_batch(args, batch)
            #input=**inputs
            # print(type(inputs))
            # flops, params = profile(model,inputs=inputs)
            # print('*'*20, 'FLOPs: ','*'*20, '\n', flops, '*'*20, 'params: ','*'*20, '\n', params, '\n')
            # print('flops: ', flops, 'params: ', params)
            # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

            # print('*'*20, '模型参数分析:', '*'*20, '\n', summary(model, inputs, show_input=False, show_hierarchical=False), '\n')    
            logits,node_logits = model(**inputs)

            graph_loss = F.cross_entropy(logits, labels)
            node_labels=torch.flatten(node_labels,0,1)
            node_logits=torch.flatten(node_logits,0,1)
            node_loss = F.cross_entropy(node_logits, node_labels,ignore_index=-1)
            loss=args.loss_weight*graph_loss+(1-args.loss_weight)*node_loss


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, dev_dataset, model)
                    test_results, test_loss = evaluate_test(args, test_dataset, model)
                    #nni.report_intermediate_result(test_results)
                    if eval_loss < eval_best_loss:
                        eval_best_loss = eval_loss
                        torch.save(model.state_dict(), args.save_model_eval_loss)

                    if results['f1_macro'] > eval_f1:
                        eval_f1 = results['f1_macro']
                        torch.save(model.state_dict(), args.save_model)


                    all_eval_results.append(results)
                    all_test_results.append(test_results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)

                    for key, value in test_results.items():
                        tb_writer.add_scalar(
                            'test_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('test_loss', test_loss, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results,all_test_results

def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels,node_labels = get_input_from_batch(args, batch)
            
            logits,node_logits = model(**inputs)
            graph_loss = F.cross_entropy(logits, labels)
            node_labels=torch.flatten(node_labels,0,1)
            node_logits=torch.flatten(node_logits,0,1)
            node_loss = F.cross_entropy(node_logits, node_labels,ignore_index=-1)
            tmp_eval_loss=graph_loss

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    return results, eval_loss


def evaluate_test(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels,node_labels = get_input_from_batch(args, batch)
            
            logits,node_logits = model(**inputs)

            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)
    return results, eval_loss

def test(args, model, test_dataset):
    model.load_state_dict(torch.load(args.save_model))
    model.eval()
    results,loss = test_detail(args, test_dataset, model)
    return results

def test_detail(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn,shuffle=False)

    # Test
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    feature_list=None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels,node_labels = get_input_from_batch(args, batch)
            
            #print('labels=',labels)
            logits,node_logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        #节点feature保存
        # if feature_list is None:
        #     feature_list=feature
        # else:
        #     feature_list=np.append(feature_list,feature,axis=0)
        #     print('feature_all_shape=',feature_list.shape)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #print('preds=',preds)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    #新加层次分类需要
    '''
    preds_list=preds.tolist()
    out_label_ids_list=out_label_ids.tolist()
    with open('true_label_1.txt','w',encoding='utf-8') as f:
        for ele in out_label_ids_list:
            ele=str(ele)+'\n'
            f.writelines(ele)
    with open('preds_1.txt','w',encoding='utf-8') as f:
        for ele_1 in preds_list:
            ele_1=str(ele_1)+'\n'
            f.writelines(ele_1)
    '''        
    #print(len(preds))
    result = compute_metrics(preds, out_label_ids)
    results.update(result)
    report=metrics.classification_report(out_label_ids,preds,target_names=['neither', 'offensive'],digits=4)
    print(report)

    output_eval_file = os.path.join(args.output_dir, 'test_results_final.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Test results *****')
        logger.info("  test loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss
    #return results

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    #acc = simple_accuracy(preds, labels)
    acc=metrics.accuracy_score(preds, labels)
    precision = metrics.precision_score(y_true=labels, y_pred=preds, pos_label=1, average='binary')
    recall = metrics.recall_score(y_true=labels, y_pred=preds, pos_label=1, average='binary')
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1_weighted=f1_score(y_true=labels, y_pred=preds, average='weighted')
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted":f1_weighted
    }

def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)