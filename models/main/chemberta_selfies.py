import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from ..ablation.model_auto_cross import Seq2SeqTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
from utils import read_delimited_file
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
from topk import topk_filter
import pandas as pd
import selfies as sf
#sed#
#loading protein smiles 
pro_smi = pd.read_json('seqID_seq_smi.json', orient='records')
torch.manual_seed(0)
#SED
s_vocab = mv.selfies_tokens_struct()

def evaluate(model, valid_iter, linear):
    model.eval()
    losses = 0
    for idx, _tgt in (enumerate(valid_iter)):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt

        tgt = _tgt.transpose(0, 1).to(device)
        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input)

        if _target is None:
            target = torch.zeros((tgt_input.size()[-1], 1024), dtype=torch.float).to(device)
        else:
            targetemb=[]
            for t in _target:
              targetemb.append(pro_smi.loc[pro_smi['seq_id'] == t]['embedding'].values[0]) ##sed##
            target = linear(torch.FloatTensor(targetemb)).to(device) ##sed##

        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_iter)


def train_epoch(model, train_iter, optimizer, linear):

    model.train()
    losses = 0
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
            target = torch.zeros((tgt_input.size()[-1], 1024), dtype=torch.float).to(device)
        else:
            targetemb=[]
            for t in _target:
              targetemb.append(pro_smi.loc[pro_smi['seq_id'] == t]['embedding'].values[0])##sed##

            target = linear(torch.FloatTensor(targetemb)).to(device)

        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)

        optimizer.zero_grad()
      
        tgt_out = tgt[1:,:]
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        losses += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, losses))
    return losses / len(train_iter)

def greedy_decode(model, max_len, start_symbol, target, linear):

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        b = 1
        s = max_len
        if target == 0:
            _target = torch.zeros((b,1024), dtype=torch.float).to(device)
        else:
            _target = linear(torch.FloatTensor(pro_smi.loc[pro_smi['seq_id'] == target]['embedding'].values[0])).to(device) ##sed##

        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, tgt_mask, _target)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1]) #[b, vocab_size]
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        #_, next_word = torch.max(prob, dim = 1)
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
    args = arg_parser.parse_args()

    print('==========  Transformer x->x ==============')



    #sed#
    #add protein smiles to vocabulary
    mol_list2 = list(pro_smi['seq_smi'].values)
    vocabulary_pro = mv.create_vocabulary(smiles_list=mol_list2, tokenizer=mv.SMILESTokenizer()) ##sed##

    ##sed##
    encoded_seqs=[]
    for sm in pro_smi['seq_smi']:
      tokens = mv.SMILESTokenizer().tokenize(sm)
      encoded = vocabulary_pro.encode(tokens)
      encoded_seqs.append(torch.tensor(encoded, dtype=torch.long))
    max_length = max([seq.size(0) for seq in encoded_seqs])
    seq_smi_emb = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
    linear=nn.Linear(in_features=max_length, out_features=1024)
    for i, seq in enumerate(encoded_seqs):
      seq_smi_emb[i, :seq.size(0)] = seq
    pro_smi['embedding']= [np.array(emb) for emb in seq_smi_emb]
    ##sed##

    mol_list0_train =list(read_delimited_file("train_selfies.csv"))[1:] 
    mol_list0_test=list(read_delimited_file("test_selfies.csv"))[1:]
    train_data = md.Dataset(mol_list0_train, s_vocab)   
    test_data  = md.Dataset(mol_list0_test, s_vocab)
    print(len(train_data),len(test_data))

    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = s_vocab.get_tokens_length()
    TGT_VOCAB_SIZE = s_vocab.get_tokens_length()

    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = 1024 

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    PAD_IDX = s_vocab.pad
    BOS_IDX = s_vocab.bos
    EOS_IDX = s_vocab.eos
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = args.device

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM, args=args)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn, drop_last=True)
    test_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_data.collate_fn, drop_last=True)
    valid_iter = test_iter

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    if args.mode == 'train':
        transformer = transformer.to(DEVICE)

        if args.loadmodel:
            transformer.load_state_dict(torch.load(args.path))

        import os

        checkpoint_path = args.path + "_checkpoint.pt"

        #loading from checkpoint if available
        start_epoch = 1
        min_loss, val_loss = float('inf'), float('inf')

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path} ...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            min_loss = checkpoint['min_loss']
            print(f"Resumed training from epoch {start_epoch} (min_loss={min_loss:.4f})")

        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer, linear)
            scheduler.step()
            end_time = time.time()
            if (epoch+1)%10==0:
                torch.save(transformer.state_dict(), args.path+'_'+str(epoch+1))
                print('Model saved every 10 epoches.') 
            
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, valid_iter, linear)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    }, args.path)
                    print('Model saved!') 
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss,
            }
            torch.save(checkpoint, checkpoint_path)

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif args.mode == 'finetune':
        from Mol_target_dataloader.utils import read_csv_file
        import Mol_target_dataloader.dataset as md

        mol_list1_train = list(read_delimited_file("HTR1A_train_sel.txt"))[1:]
        target_list1_train =  ['P08908'] * len(mol_list1_train) 

        mol_list1_val = list(read_delimited_file("HTR1A_valid_sel.txt"))[1:]
        target_list1_val = ['P08908'] * len(mol_list1_val) 

        mol_list2_train = list(read_delimited_file("DRD2_train_sel.txt"))[1:]
        target_list2_train = ['P14416'] * len(mol_list2_train) 

        mol_list2_val = list(read_delimited_file("DRD2_valid_sel.txt"))[1:]
        target_list2_val = ['P14416'] * len(mol_list2_train) 
        
        mol_list1_train.extend(mol_list2_train)
        target_list1_train.extend(target_list2_train)
        mol_list1_val.extend(mol_list2_val)
        target_list1_val.extend(target_list2_val)
        

        train_data = md.Dataset(mol_list1_train, target_list1_train, s_vocab)
        val_data = md.Dataset(mol_list1_val, target_list1_val, s_vocab)
        
        train_iter = tud.DataLoader(train_data, args.batch_size, collate_fn=train_data.collate_fn, shuffle=True)
        val_iter = tud.DataLoader(val_data, args.batch_size, collate_fn=val_data.collate_fn, shuffle=True)

        transformer = transformer.to(DEVICE)
        transformer.load_state_dict(torch.load("model_base_prosmi_dgsel.h5"))

        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer, linear)
            scheduler.step()
            end_time = time.time()
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, val_iter,linear)
                if val_loss < min_loss:
                    min_loss = val_loss
                    #torch.save(transformer.state_dict(), "model_finetune_pro_smi.h5")
                    torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    }, args.path)
                    print('Model saved!')

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))

        
    elif args.mode == 'infer':
        if args.device == 'cpu':
            checkpoint = torch.load(args.path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.path)

        # Check whether it's a full checkpoint dict or just state_dict
        if "model_state_dict" in checkpoint:
            transformer.load_state_dict(checkpoint["model_state_dict"])
        else:
            transformer.load_state_dict(checkpoint)
        device = args.device
        transformer.to(device)
        transformer.eval()
        _target = args.target
        print('Target: {0}'.format(_target))
        f=open("Results/{0}.txt".format(_target),'a')
        for i in range(10000):
            ybar = greedy_decode(transformer, max_len=100, start_symbol=s_vocab.bos, target=_target, linear=linear).flatten()
            decoded_selfies = s_vocab.decode(ybar.to('cpu').data.numpy())
            try:
                decoded_smiles = sf.decoder(decoded_selfies)  # selfies → smiles
            except:
                decoded_smiles = ""
            f.write(decoded_smiles + "\n")
            print(i)
        f.close() 
       
