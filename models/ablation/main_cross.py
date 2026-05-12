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
from model_auto_cross import Seq2SeqTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
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

# === unified checkpoint helpers ===
def save_checkpoint(checkpoint, filename="checkpoint_last.pth"):
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None, map_location=None, strict=True):
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'],strict=strict)
    if optimizer is not None and checkpoint.get('optimizer') is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return model, optimizer, scheduler, start_epoch, best_val_loss

#loading protein embeddings for teacher
pro_emb = pd.read_json('seqID_embedding.json', orient='records') 
pro_emb['embedding'] = pro_emb['embedding'].apply(np.array)
seqid_to_emb = dict(zip(pro_emb['seq_id'], pro_emb['embedding']))
torch.manual_seed(0)

def evaluate(model, valid_iter, device, epoch, EMB_SIZE, loss_fn=None):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in valid_iter:
            _drg = batch
            _target = None
            if isinstance(batch, tuple):
                _drg, _target = batch

            drg = _drg.transpose(0, 1).to(device)
            drg_input = drg[:-1, :]
            drg_mask, drg_padding_mask = create_mask(drg_input)

            if _target is None:
                target_stu = torch.zeros((drg_input.size(1), EMB_SIZE), dtype=torch.float).to(device)
            else:
                targetemb_stu=[]
                for t in _target:
                  targetemb_stu.append(seqid_to_emb[t])
                target_stu = torch.FloatTensor(np.stack(targetemb_stu)).to(device)

            logits = model(drg_input, drg_mask, drg_padding_mask, target_stu)
            drg_out = drg[1:, :]
            
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = drg_out.reshape(-1)
            ce_loss = loss_fn(logits_flat, labels_flat) 
            loss = ce_loss 
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches

    return avg_loss

def train_epoch(model, train_iter, optimizer, device, epoch, EMB_SIZE, loss_fn=None):
    model.train()
    total_loss = 0
    for idx, _drg in enumerate(train_iter):
        _target = None
        if type(_drg) is tuple:
            _drg, _target = _drg

        drg = _drg.transpose(0, 1).to(device)
        drg_input = drg[:-1, :]
        drg_mask, drg_padding_mask = create_mask(drg_input)

        if _target is None:
            target_stu = torch.zeros((drg_input.size(1), EMB_SIZE), dtype=torch.float).to(device)
        else:
            targetemb_stu=[]
            for t in _target:
              targetemb_stu.append(seqid_to_emb[t])

            target_stu = torch.FloatTensor(np.stack(targetemb_stu)).to(device)

        logits = model(drg_input, drg_mask, drg_padding_mask, target_stu) 
        optimizer.zero_grad()
        drg_out = drg[1:,:]
        ce_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), drg_out.reshape(-1)) 
        
        loss = ce_loss 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if idx % 100 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        total_loss += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, total_loss))

    return total_loss / len(train_iter)
 
def greedy_decode(model, max_len, k, start_symbol, EOS_IDX, target, device, EMB_SIZE):

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    b = 1
    if target == "0":
        _target = torch.zeros((b, EMB_SIZE), dtype=torch.float).to(device)
    else:
        _target = torch.FloatTensor(seqid_to_emb[target]).unsqueeze(0).to(device)
    for i in range(max_len-1):
        
        drg_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        drg_padding_mask = (ys == PAD_IDX).transpose(0, 1)
        
        # forward pass
        logits = model(ys, drg_mask, drg_padding_mask, _target)
        # take last step
        logits = logits[-1, :, :]   # (B=1, vocab)
        logits = topk_filter(logits, top_k=30)
        probs = F.softmax(logits, dim=-1)
        
        next_word = torch.multinomial(probs, 1).item()
        
        # append
        ys = torch.cat(
            [ys, torch.ones(1, 1).fill_(next_word).long().to(device)],
            dim=0
        )
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
    arg_parser.add_argument('--FFN', default=1024, type=int)
    arg_parser.add_argument('--nhead', default=8, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    arg_parser.add_argument('--num', default=10000, type=int)
    arg_parser.add_argument('--inferpath', default='Results/', type=str)
    arg_parser.add_argument('--topk', default=30, type=int)
    args = arg_parser.parse_args()
    
    print('==========  Transformer x->x ==============')
    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = args.FFN 

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    
    vocabulary = mv.tokens_struct()
    PAD_IDX = vocabulary.pad
    BOS_IDX = vocabulary.bos
    EOS_IDX = vocabulary.eos
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = vocabulary.get_tokens_length()
    drg_VOCAB_SIZE = vocabulary.get_tokens_length()
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, drg_VOCAB_SIZE,
                                 FFN_HID_DIM, args=args)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    #lr=0.0001
    lr=1e-6
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    patience=12
    
    if args.mode == 'train':
        mol_list_train = list(read_delimited_file('train.smi'))
        mol_list_val = list(read_delimited_file('valid.smi'))
        
        print("sampling 20 percent of data.")

        np.random.seed(42)  # عدد ثابت
        mol_list_train = np.random.choice(mol_list_train, size=int(0.2 * len(mol_list_train)), replace=False)
        mol_list_val = np.random.choice(mol_list_val, size=int(0.2 * len(mol_list_val)), replace=False)
        
        train_data = md.Dataset(mol_list_train, None, vocabulary)
        test_data = md.Dataset(mol_list_val, None, vocabulary)
        train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn, drop_last=True, num_workers=4, pin_memory=True)
        valid_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_data.collate_fn, drop_last=True, num_workers=4, pin_memory=True)
    
        transformer = transformer.to(DEVICE)

        min_loss, val_loss = float('inf'), float('inf')
        epochs_no_improve = 0
        start_epoch =1
        if args.loadmodel:
            print("loading checkpoint")
            transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                args.path, transformer, optimizer, scheduler
            )
            min_loss=best_val_loss
            
        
        best_checkpoint = None
        last_checkpoint = None
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            try:    
                start_time = time.time()
                train_loss = train_epoch(transformer, train_iter, optimizer, DEVICE, epoch, EMB_SIZE, loss_fn)
                scheduler.step()
                end_time = time.time()
            
                val_loss = evaluate(transformer, valid_iter, DEVICE,epoch, EMB_SIZE, loss_fn)
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_checkpoint = {
                    'epoch': epoch,
                    'state_dict': transformer.state_dict(),
                    'optimizer': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'best_val_loss': min_loss
                    }
                    epochs_no_improve = 0
                else:
                  epochs_no_improve +=1
                  print(f"no improvement for { epochs_no_improve } epochs")
                  
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted during training step!")
                if best_checkpoint is not None:
                    save_checkpoint(best_checkpoint, filename=f"{args.path}_best.pth")
                if last_checkpoint is not None:   
                    torch.save(last_checkpoint, filename=f"{args.path}_last.pth")
                print(f"Checkpoint saved at {filename}")
                exit(0)
            
            last_checkpoint={
                    'epoch': epoch,
                    'state_dict': transformer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
              
            if best_checkpoint is not None and epoch % 10 ==0:
                save_checkpoint(best_checkpoint, filename=f"{args.path}_best.pth")
                print('Saved best model (val loss)')
            
            if patience <= epochs_no_improve:
              print("Early stopping triggered.")
              break
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
        if best_checkpoint is not None:
            save_checkpoint(best_checkpoint, filename=f"{args.path}_best.pth")
            print('Saved best model (val loss)')

    elif args.mode == 'finetune':
       
        mol_list_train = list(read_delimited_file('HTR1A_train.txt'))
        target_list_train = ['P08908'] * len(mol_list_train) 

        mol_list_val = list(read_delimited_file('HTR1A_valid.txt'))
        target_list_val = ['P08908'] * len(mol_list_val) 

        mol_list2_train = list(read_delimited_file('DRD2_train.txt'))
        target_list2_train = ['P14416'] * len(mol_list2_train) 

        mol_list2_val = list(read_delimited_file('DRD2_valid.txt'))
        target_list2_val = ['P14416'] * len(mol_list2_val) 
        
        
        mol_list_train.extend(mol_list2_train)
        target_list_train.extend(target_list2_train)
        mol_list_val.extend(mol_list2_val)
        target_list_val.extend(target_list2_val)
        

        train_data = md.Dataset(mol_list_train, target_list_train, vocabulary)
        val_data = md.Dataset(mol_list_val, target_list_val, vocabulary)
        train_iter = tud.DataLoader(train_data, args.batch_size, collate_fn=train_data.collate_fn, shuffle=True)
        val_iter = tud.DataLoader(val_data, args.batch_size, collate_fn=val_data.collate_fn, shuffle=True)
        
        transformer = transformer.to(DEVICE)
        transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                "base_1024_sharedvocab_layers_best.pth", model=transformer,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=DEVICE,
                strict=False
            )
        

        # === checkpoint start ===
        min_loss, val_loss = float('inf'), float('inf')
        epochs_no_improve = 0
        start_epoch = 1
        if args.loadmodel:
            print("loading checkpoint")
            transformer, optimizer, scheduler, start_epoch, best_val_loss= load_checkpoint(
                args.path, transformer
            )
            min_loss=best_val_loss
    
        
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer, DEVICE, epoch,EMB_SIZE, loss_fn=loss_fn)

            scheduler.step()
            end_time = time.time()

            val_loss = evaluate(transformer, val_iter, DEVICE,epoch, EMB_SIZE, loss_fn=loss_fn)
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
                epochs_no_improve = 0
                print('Saved best model (val loss)')
            else:
              epochs_no_improve +=1
              print(f"no improvement for { epochs_no_improve } epochs")
            if patience <= epochs_no_improve:
              print("Early stopping triggered.")
              break
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
        gen_smi=[]
        for i in range(args.num):
            ybar = greedy_decode(transformer, max_len=100, k=args.topk, start_symbol=BOS_IDX, EOS_IDX=EOS_IDX, target=_target, device=DEVICE, EMB_SIZE=EMB_SIZE).flatten()
            ybar = vocabulary.decode(ybar.to('cpu').data.numpy())
            gen_smi.append(ybar)
        gen=pd.DataFrame(gen_smi)
        gen.to_csv(args.inferpath+"/{0}.txt".format(_target),header=None,index=False)
       

    
