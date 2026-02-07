import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from model_auto import Seq2SeqTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import dataset as md
import torch.utils.data as tud
from utils import read_delimited_file
import os.path
import glob
from collections import Counter
from torch import Tensor
import io
import time
from topk import topk_filter
import wandb

# === unified checkpoint helpers ===
def save_checkpoint(checkpoint, filename="checkpoint_last.pth"):
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None, map_location=None):
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and checkpoint.get('optimizer') is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return model, optimizer, scheduler, start_epoch, best_val_loss

pro_smi = pd.read_pickle("seqID_seq_smi_with_emb.pkl")
seqid_to_emb = dict(zip(pro_smi['seq_id'], pro_smi['chemberta_emb']))
torch.manual_seed(0)

def evaluate(model, valid_iter, linear, vocabulary, tokenizer, device, epoch, EMB_SIZE, loss_fn=None):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in valid_iter:
            _tgt = batch
            _target = None
            if isinstance(batch, tuple):
                _tgt, _target = batch

            tgt = _tgt.transpose(0, 1).to(device)
            tgt_input = tgt[:-1, :]
            tgt_mask, tgt_padding_mask = create_mask(tgt_input)

            if _target is None:
                target_stu = torch.zeros((tgt_input.size(1), 1024), dtype=torch.float).to(device)
            else:
                targetemb_stu=[]
                for t in _target:
                  targetemb_stu.append(seqid_to_emb[t])
                target_stu = linear(torch.FloatTensor(np.stack(targetemb_stu))).to(device) ##sed##

            logits = model(tgt_input, tgt_mask, tgt_padding_mask, target_stu)
            tgt_out = tgt[1:, :]
            
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = tgt_out.reshape(-1)
            ce_loss = loss_fn(logits_flat, labels_flat) 
            loss = ce_loss 
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    wandb.log({
        "val_loss": total_loss / n_batches,
        "epoch": epoch
    })
    return avg_loss
    
def train_epoch(model, train_iter, optimizer, linear, vocabulary, tokenizer, device, epoch, EMB_SIZE, loss_fn=None):

    model.train()
    total_loss = 0
    for idx, _tgt in enumerate(train_iter):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt
            #_target = torch.LongTensor(_target).to(device)

        #print(type(_tgt) is tuple)
        tgt = _tgt.transpose(0, 1).to(device)
        # remove encoder
        tgt_input = tgt[:-1, :]
        tgt_mask, tgt_padding_mask = create_mask(tgt_input)

        if _target is None:
            target_stu = torch.zeros((tgt_input.size(1), EMB_SIZE), dtype=torch.float).to(device)
        else:
            targetemb_stu=[]
            for t in _target:
              targetemb_stu.append(seqid_to_emb[t])

            target_stu = linear(torch.FloatTensor(np.stack((targetemb_stu)))).to(device)

        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target_stu) 
        optimizer.zero_grad()
        tgt_out = tgt[1:,:]
        ce_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) 
        
        loss = ce_loss 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if idx % 100 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        total_loss += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, total_loss))
    wandb.log({
        "train_loss": total_loss / len(train_iter),
        "epoch": epoch
    })
    return total_loss / len(train_iter)

def greedy_decode(model, max_len, start_symbol, EOS_IDX, target, linear, device, EMB_SIZE):

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        b = 1
        s = max_len
        if target == "0":
            _target = torch.zeros((b, EMB_SIZE), dtype=torch.int32).to(device)
        else:
            _target = torch.FloatTensor(seqid_to_emb[target]).unsqueeze(0).to(device)
            _target = linear(_target)

        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, tgt_mask, _target)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1]) #[b, vocab_size]
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
          break
    return ys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer', 'baseline', 'finetune'], default='finetune',help='Run mode')
    arg_parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',help='Device')
    arg_parser.add_argument('--epoch', default='100', type=int)
    arg_parser.add_argument('--batch_size', default='512', type=int)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--path', default='model_base.h5', type=str)
    arg_parser.add_argument('--datamode', default=1, type=int)
    arg_parser.add_argument('--target', default=0, type=str)
    arg_parser.add_argument('--d_model', default=1024, type=int)
    arg_parser.add_argument('--nhead', default=8, type=int)
    arg_parser.add_argument('--embedding_size', default=200, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    arg_parser.add_argument('--num', default=10000, type=int)
    args = arg_parser.parse_args()
    
    print('==========  Transformer x->x ==============')
    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = 1024 

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mol_list0_train = list(read_delimited_file('train.smi'))
    mol_list0_test = list(read_delimited_file('test.smi'))
    

    mol_list1 = list(read_delimited_file('HTR1A_train.txt'))
    mol_list1.extend(list(read_delimited_file('HTR1A_test.txt')))
    mol_list1.extend(list(read_delimited_file('HTR1A_valid.txt')))

    mol_list1.extend(list(read_delimited_file('DRD2_train.txt')))
    mol_list1.extend(list(read_delimited_file('DRD2_test.txt')))
    mol_list1.extend(list(read_delimited_file('DRD2_valid.txt')))

    mol_list = mol_list0_train
    mol_list.extend(mol_list0_test) 
    mol_list.extend(mol_list1)
    vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())

    linear=nn.Linear(in_features=768, out_features=1024) ## protein smiles chemberta embedding
    linear = linear.to(DEVICE)
 
    train_data = md.Dataset(mol_list0_train, vocabulary, mv.SMILESTokenizer())
    test_data = md.Dataset(mol_list0_test, vocabulary, mv.SMILESTokenizer())

    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = len(vocabulary)
    TGT_VOCAB_SIZE = len(vocabulary)

    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM, args=args)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn, drop_last=True)
    test_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_data.collate_fn, drop_last=True)
    valid_iter = test_iter
    lr=0.0001
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    
    wandb.init(
    project="drug-protein-generation",  
    name="exp2_linChemberta_pro_smi_dg_smi",                        
    config={
        "learning_rate": lr,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
    }
    )
    config = wandb.config

    if args.mode == 'train':
        transformer = transformer.to(DEVICE)

        min_loss, val_loss = float('inf'), float('inf')
        if args.loadmodel:
            print("loading checkpoint")
            transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                args.path, transformer, optimizer, scheduler
            )
            min_loss=best_val_loss
            
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer, linear, vocabulary, mv.SMILESTokenizer(), DEVICE, epoch, loss_fn)
            scheduler.step()
            end_time = time.time()
            
            val_loss = evaluate(transformer, val_iter, linear, vocabulary, mv.SMILESTokenizer(), DEVICE,epoch, imagedta_model, loss_fn)
            if val_loss < min_loss:
                min_loss = val_loss
                checkpoint = {
                'epoch': epoch,
                'state_dict': transformer.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': min_loss
                }
                save_checkpoint(checkpoint, filename=f"{args.path}_best.pth")
                print('Model saved!') 

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif args.mode == 'finetune':
        from Mol_target_dataloader.utils import read_csv_file
        import Mol_target_dataloader.dataset as md

        #sed#
        
        mol_list1_train = list(read_delimited_file('HTR1A_train.txt'))
        target_list1_train = ['P19327'] * len(mol_list1_train) + ['P08908'] * len(mol_list1_train) + ['Q64264'] * len(mol_list1_train)
        mol_list1_train = mol_list1_train * 3

        mol_list1_val = list(read_delimited_file('HTR1A_valid.txt'))
        target_list1_val = ['P19327'] * len(mol_list1_val) + ['P08908'] * len(mol_list1_val) + ['Q64264'] * len(mol_list1_val)
        mol_list1_val = mol_list1_val * 3

        mol_list2_train = list(read_delimited_file('DRD2_train.txt'))
        target_list2_train = ['P14416'] * len(mol_list2_train) + ['P61169'] * len(mol_list2_train) + ['P61168'] * len(mol_list2_train)
        mol_list2_train = mol_list2_train * 3

        mol_list2_val = list(read_delimited_file('DRD2_valid.txt'))
        target_list2_val = ['P14416'] * len(mol_list2_val) + ['P61169'] * len(mol_list2_val) + ['P61168'] * len(mol_list2_val)
        mol_list2_val = mol_list2_val * 3
        
        mol_list1_train.extend(mol_list2_train)
        target_list1_train.extend(target_list2_train)
        mol_list1_val.extend(mol_list2_val)
        target_list1_val.extend(target_list2_val)
        

        train_data = md.Dataset(mol_list1_train, target_list1_train, vocabulary, mv.SMILESTokenizer())
        val_data = md.Dataset(mol_list1_val, target_list1_val, vocabulary, mv.SMILESTokenizer())
        #sed#
        train_iter = tud.DataLoader(train_data, args.batch_size, collate_fn=train_data.collate_fn, shuffle=True, num_workers=0)
        val_iter = tud.DataLoader(val_data, args.batch_size, collate_fn=val_data.collate_fn, shuffle=True, num_workers=0)

        transformer = transformer.to(DEVICE)
        transformer.load_state_dict(torch.load('model_base_1024.h5'))
        # === checkpoint start ===
        min_loss, val_loss = float('inf'), float('inf')
        start_epoch = 1
        if args.loadmodel:
            print("loading checkpoint")
            transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                args.path, transformer
            )
            min_loss=best_val_loss

        imagedta_model = get_imagedta_model(device=DEVICE)
    
        wandb.watch(transformer, log="all", log_freq=100)
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer, linear, vocabulary, mv.SMILESTokenizer(), DEVICE, epoch, imagedta_model, loss_fn=loss_fn)

            scheduler.step()
            end_time = time.time()

            val_loss = evaluate(transformer, val_iter, linear, vocabulary, mv.SMILESTokenizer(), DEVICE,epoch, imagedta_model, loss_fn=loss_fn)
            if val_loss < min_loss:
                min_loss = val_loss
                checkpoint = {
                'epoch': epoch,
                'state_dict': transformer.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': min_loss
                }
                save_checkpoint(checkpoint, filename=f"{args.path}_best.pth")
                print('Saved best model (val loss)')
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))

        
    elif args.mode == 'infer':
        if args.device == 'cpu':
            transformer.load_state_dict(torch.load(args.path,  map_location=torch.device('cpu')))
        else:
            transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                args.path, transformer
            )
        transformer.to(DEVICE)
        transformer.eval()
        _target = args.target
        print('Target: {0}'.format(_target))
        f=open("Results/T_"+str(T)+"_alpha_"+str(alpha)+"/{0}.txt".format(_target),'a')
        for i in range(args.num):
            ybar = greedy_decode(transformer, max_len=100, start_symbol=BOS_IDX, target=_target, linear=linear, device=DEVICE).flatten()
            #print(ybar)
            ybar = mv.SMILESTokenizer().untokenize(vocabulary.decode(ybar.to('cpu').data.numpy()))
            #print('prediction')
            f.write(ybar+"\n")
            print(i)
        f.close() 
       
