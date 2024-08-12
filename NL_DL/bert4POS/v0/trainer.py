"""
  TODO:
    - load checkpoint
"""
GCP = False     # running on google cloud?

import os
import sys
from enum import Enum
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.optim import AdamW
import logging
import json
import matplotlib.pyplot as plt


BERTDIR = ''        # where bert model and tokenizer live
UDDIR = ''          # location of the universal dependencies dataset
OUTDIR = ''         # where output should go

if GCP:
    import google.cloud.logging
    client = google.cloud.logging.Client()
    client.setup_logging()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# torch.manual_seed(13)
POS = Enum('POS', ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                   'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM',
                   'VERB', 'X'])

defaultBertConfig = {'bertDir': BERTDIR,
                    'hidden_size': 18,
                    'num_hidden_layers' : 1,
                    'num_attention_heads' : 2,
                    'intermediate_size' : 32,}

defaultTrainConfig = {'maxTokens': 128,
                      'nEpochs': 4,
                      'lr': 1e-3,
                      'trainBatchSize': 16,
                      'maskProp': 0.2,
                      'udDir': UDDIR,
                      'trainFile':  '20kTrain.conllu', # very small training file
                      'validFile':  '10kDev.conllu',
                      'testFile': 'en_ewt-ud-test.conllu',
                      'outDir': OUTDIR,
                      'ckptFname': 'ckpt.pt',
                      'lossesFile': 'losses.csv',
                      'learningCurve': 'learningCurve.png',
                      'savedTrainConfig': 'trainConfig.jsom',
                      'savedBertConfig': 'bertConfig.json',
                      }

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print('DEVICE: ', DEVICE)

tokenizer = None

def init(trainConfig, bertConfig):
    global tokenizer
    tokenizerLoc = os.path.join(bertConfig['bertDir'], 'tokenizer')
    print('tokeniser: ', tokenizerLoc)
    tokenizer = BertTokenizer.from_pretrained(tokenizerLoc,
        model_max_length=trainConfig['maxTokens'])
    if bertConfig['hidden_size'] < len(POS):
        raise ValueError('Hidden size should be > num features')

#init(defaultTrainConfig, defaultBertConfig)

def makeSentenceGen(tvt: str, trainConfig: dict):
    """
    Generates one sentence at a time from the UD data
      - assume sentene indexes start from 1
      - ignore range indexes
    :param tvt: the type of file: trainFile / validFile / testFile
    :yield: list of [word, POS]
    """
    inFile = open(os.path.join(trainConfig['udDir'], trainConfig[tvt]), 'r')
    df = pd.read_csv(inFile,
                     comment='#', index_col=None, header=None,
                     sep='\t', usecols=[0, 1, 3])
    # df[0] = pd.to_numeric(df[0])
    idx = 0
    sentence = []
    while True:
        if idx >= len(df):
            if len(sentence) > 0:
                yield sentence
                sentence = []
            else:
                logging.info('File finished')
                inFile.close()
                raise StopIteration
        try:
            if '-' not in df.iloc[idx, 0] and ':' not in df.iloc[idx, 0]:
                widx = int(df.iloc[idx, 0])
                if widx == 1:
                    if len(sentence) > 0:
                        yield sentence
                        sentence = []
                    sentence = [[df.iloc[idx, 1], POS[df.iloc[idx, 2]]]]
                else:
                    sentence.append([df.iloc[idx, 1], POS[df.iloc[idx, 2]]])
        except Exception as e:
            pass
            # print('ERROR:\n', df.iloc[idx])
            # logging.error('Error in row %d\n%s' % (idx, str(e)))
        idx += 1

def batchGen(tvt: str, trainConfig: dict, bertConfig: dict):
    """
    generates masked batches of text
    :param tvt: which split to process: trainFile/testFile/devFile
    :param trainConfig: parameters
    :yields: batch of sentences as: encodings, true tensor, mask
    """
    sentenceGen = makeSentenceGen(tvt, trainConfig)
    sentences = []
    while True:
        try:
            sentences.append(next(sentenceGen))
        except StopIteration:
            logging.info('No more batches')
            raise
        if len(sentences) == trainConfig['trainBatchSize']:
            encoding, trues, mask = tokenize(sentences, trainConfig, bertConfig)
            yield encoding, trues, mask
            sentences = []

def tokenize(sentEmb,  # batch of list of [word, POS]
            trainConfig,
             bertConfig):
    # returns BatchEncoding for the sentences, target tensor, mask
    sentences = []
    for s in sentEmb:
        sentences.append(' '.join([x[0] for x in s]))
    embeddings = []
    embs = []
    noneEmb = torch.zeros((3))
    be = tokenizer(sentences, padding=True, truncation=True,
                   max_length=trainConfig['maxTokens'],
                   return_tensors='pt')
    for i, ids in enumerate(be.input_ids):  # each sentence
        posList = [x[1] for x in sentEmb[i]]
        thisEmb = []
        cntr = 0
        for j, tid in enumerate(ids):       # each token
            tok = tokenizer.decode([tid])
            if tok.startswith('#'):         # same POS for all wordpieces of a word
                thisEmb.append(thisEmb[-1])
            elif cntr < len(posList) and tok in sentences[i]:
                thisEmb.append(getRep(posList[cntr], bertConfig))
                cntr += 1
            else:
                # print('tokenize adding none')
                thisEmb.append(getRep(POS['X'], bertConfig))
        xx = torch.stack(thisEmb, dim=0)
        embs.append(xx)
    rc = torch.stack(embs)
    be, mask = applyMask(be, tokenizer, trainConfig)
    return be, rc, mask

def getRep(pos, bertConfig):
    # use one hot encoding for POS
    rv = torch.zeros(bertConfig['hidden_size'])
    rv[pos.value - 1] = 1
    return rv

def applyMask(be, tk, trainConfig):
    """
    given a batch encoding and the targets:
      - for each row:
        - apply a mask to the  input_ids and set the token to mask_token
      - return be and mask
    """
    iid = be.input_ids
    size = iid.shape[:2]
    # dont want to mask CLS or SEP or PAD
    mask = torch.where((iid != tk.cls_token_id) * (iid != tk.sep_token_id) * (iid != tk.pad_token_id),
                       1.0, 0.0) * torch.rand(size)
    mask = torch.where((mask > 0) * (mask < trainConfig['maskProp']), 1.0, 0.0)
    for row in range(iid.shape[0]):
        toMask = torch.flatten(mask[row].nonzero()).tolist()
        iid[row, toMask] = tk.mask_token_id
    return be, mask

def trainOneEpoch(model, optim, ceLoss, trainGen, trainConfig, epoch):
    bcnt = 0
    sumLoss = 0
    while True:
        try:
            encoding, trues, mask = next(trainGen)
            bcnt += 1
        except StopIteration:
            break
        except Exception as e:
            logging.error('trainOneEPoch ' + str(e))
        optim.zero_grad()
        encoding = encoding.to(DEVICE)
        mask = mask.to(DEVICE)
        trues = trues.to(DEVICE)
        preds = model(**encoding)
        mpreds = torch.mul(preds.last_hidden_state, mask.unsqueeze(-1))
        # print('mpreds ', mpreds.shape)
        mtarget = torch.mul(trues, mask.unsqueeze(-1))
        # print('mtrget ', mtarget.shape)
        loss = ceLoss(mpreds, mtarget)
        loss.backward()
        sumLoss += loss.item()
        optim.step()
        if bcnt % 100 == 0:
            print('Train epoch %d, batch %d, loss: %f' % (epoch, bcnt, loss/trainConfig['trainBatchSize']))
    return sumLoss/(bcnt)

def evalModel(model, ceLoss, valGen, trainConfig, epoch):
    bcnt = 0
    sumLoss = 0
    while True:
        try:
            encoding, trues, mask = next(valGen)
            bcnt += 1
        except StopIteration:
            break
        except Exception as e:
            logging.error('evalModel ' + str(e))
        encoding = encoding.to(DEVICE)
        mask = mask.to(DEVICE)
        trues = trues.to(DEVICE)
        preds = model(**encoding)
        mpreds = torch.mul(preds.last_hidden_state, mask.unsqueeze(-1))
        mtarget = torch.mul(trues, mask.unsqueeze(-1))
        loss = ceLoss(mpreds, mtarget)
        sumLoss += loss.item()
        if bcnt % 100 == 0:
            print('Eval epoch %d, batch %d, loss: %f' % (epoch, bcnt, loss / trainConfig['trainBatchSize']))
    return sumLoss / (bcnt)

def train(trainConfig, bertConfig):
    init(trainConfig, bertConfig)
    os.makedirs(trainConfig['outDir'], exist_ok=True)
    bcObj = BertConfig(**bertConfig)
    model = BertModel(bcObj)
    model.to(DEVICE)
    model.train()
    optim = AdamW(model.parameters(), lr = trainConfig['lr'])
    loss = nn.CrossEntropyLoss()
    minLoss = 1e9
    trainLosses = []
    valLosses = []
    for i in range(trainConfig['nEpochs']):
        print('-------------epoch ', i)
        model.train()
        trainGen = batchGen('trainFile', trainConfig, bertConfig)
        try:
            trainLoss = trainOneEpoch(model, optim, loss, trainGen, trainConfig, i)
        except:
            print('Error trainOneEPoch epoch %d: %s ', (i, str(e)))
        trainLosses.append(trainLoss)
        model.eval()
        valGen = batchGen('validFile', trainConfig, bertConfig)
        valLoss = evalModel(model, loss, valGen, trainConfig, i)
        valLosses.append(valLoss)
        print('Epoch %d, train loss %f, valLoss %f' % (i, trainLoss, valLoss))
        logging.info('Epoch %d, train loss %f, valLoss %f' % (i, trainLoss, valLoss))
        if valLoss < minLoss:
            minLoss = valLoss
            ckptFile = os.path.join(trainConfig['outDir'], trainConfig['ckptFname'])
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': i},
                       ckptFile)
    df = pd.DataFrame(zip(trainLosses, valLosses), columns=['train', 'val'])
    df.to_csv(os.path.join(trainConfig['outDir'], trainConfig['lossesFile']),
              index=False)
    plt.plot(range(len(trainLosses)), trainLosses)
    plt.plot(range(len(valLosses)), valLosses)
    plt.savefig(os.path.join(trainConfig['outDir'], trainConfig['learningCurve']))
    with open(os.path.join(trainConfig['outDir'], trainConfig['savedTrainConfig']), 'w') as jx:
        json.dump(trainConfig, jx, indent = 4)
    with open(os.path.join(trainConfig['outDir'], trainConfig['savedBertConfig']), 'w') as jx:
        json.dump(bertConfig, jx, indent=4)

