import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation

# load train, validation, test, vocab
data = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/dev_fixtures/prediction.npz')
fixtures_gen = np.load('../fixtures/dev_fixtures/generation.npy')
fixtures_pred_test = np.load('../fixtures/test_fixtures/prediction.npz')
fixtures_gen_test = np.load('../fixtures/test_fixtures/generation.npy')
vocab = np.load('../dataset/vocab.npy')

def collate(seq_list):
    """Transforms a list of sequences into a batch. 
    Passed to DataLoader, returns data shaped as L x B
    Takes in:
    - list of sequences of size N x 2 x L
    Returns:
    - list of sequences of size L x B as inputs
    - list of sequences of size L x B as targets
    """
    inputs = torch.cat([s[0].unsqueeze(1) for s in seq_list],dim=1)
    targets = torch.cat([s[1].unsqueeze(1) for s in seq_list],dim=1)
    return inputs,targets

class FixedSequenceDataset(Dataset):
    def __init__(self, articles, seq_len):
        """Fixed Sequence Dataset.
        Our training data contains 579 articles, each an array of 
        words of different lengths. We want to create a dataset from
        that data by splitting these articles into chunks of fixed length.

        We return each chunk as data and as target - with the target 
        behind ahead of the data by one instance at each time.

        L: sequence length
        M: batch size
        """
        self.articles = articles
        self.seq_len = seq_len
        self._reload_data()
    
    def _reload_data(self, seq_len=None):
        """Reload the data.
        Call this method before every training epoch. This
        will re-shuffle the training data into new sequences
        of a (potentially different) sequence length.
        No output.
        """
        self.seq_len = seq_len if seq_len else self.seq_len
        random.shuffle(self.articles)
        text = np.concatenate(self.articles).ravel()
        cutoff = (len(text) // self.seq_len) * self.seq_len
        text = text[:cutoff]
        self.data = torch.tensor(text, dtype=torch.int).view(-1, self.seq_len)
        self.data = self.data.type(torch.LongTensor)
    
    def __getitem__(self,i):
        """
        Returns data point at index i as:
        - tensor of size L as input
        - tensor of size L as target
        """
        text = self.data[i]
        return text[:-1], text[1:]
    
    def __len__(self):
        return self.data.size(0)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout_input=0.5, dropout_hidden=0.5, 
                dropout_embed=0.1, dropout_final=0.4, weight_drop=0.1, tie_weights=False):
        """Language model.
        Consists of:
        - an embedding layer
        - a three-layer LSTM
        - a linear layer for scoring
        Notes:
        - Changing weight dropped LSTM with 0.3 instead of 0.5
        - TODO: dropout on word vectors of 0.4
        """
        super(LanguageModel, self).__init__()

        # save parameters
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.weight_drop = weight_drop
        self.tie_weights = tie_weights
        self.dropout_input = dropout_input
        self.dropout_embed = dropout_embed
        self.dropout_hidden = dropout_hidden
        self.dropout_final = dropout_final

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.recurrent = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout_hidden)
        self.scoring = nn.Linear(hidden_size, vocab_size)
        # alternative to nlp_nn.WeightDropLSTM: LSTM + WeightDrop
        #self.recurrent = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers)
        #self.drop_connect = nlp_nn.WeightDrop(self.recurrent, ['weight_hh'], dropout=0.9)
        self.inp_dropout = nn.Dropout(dropout_input)
        self.emb_dropout = nn.Dropout(dropout_embed)
        self.final_dropout = nn.Dropout(dropout_final)

        # initialize the weights
        self.init_weights()
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.scoring.bias.data.fill_(0)
        self.scoring.weight.data.uniform_(-0.1, 0.1)

    def forward(self, seq_batch):
        """Forward pass in the model.
        Takes:
        - seq_batch: dim L x B
        Returns:
        - output: dim L x B x V
        """
        # L: sequence length
        # B: batch size
        # E: embed size
        # H: hidden size
        # V: vocab size
        
        # input of size L x B
        batch_size = seq_batch.size(1)
        # embedding and dropout
        embed = self.embedding(seq_batch)
        embed = self.emb_dropout(embed)
        # LSTM - embedded data as L x B x E
        # output of LSTM - L x B x H
        hidden = None
        output_lstm, hidden = self.recurrent(embed, hidden)
        output_lstm = self.final_dropout(output_lstm)
        # flatten after LSTM to (L*B) x H
        output_lstm_flatten = output_lstm.view(-1, self.hidden_size)
        # full linear layer for scoring - to (L*B) * V
        output_scoring = self.scoring(output_lstm_flatten)
        # return as L x B x V
        return output_scoring.view(-1,batch_size,self.vocab_size)
    
    def generate(self, seq, n_words):
        """Generate word (sequence) from given sequence.
        Takes:
        - seq: dim L x B
        - n_words: int
        Returns:
        - word_sequence: list of length n_words
        """
        # perform greedy search to extract and return words (one sequence).
        generated_words = []
        # feed it into the network similar to the forward pass
        # but with batch size being 1
        embed = self.embedding(seq)
        embed = self.emb_dropout(embed)
        hidden = None
        output_lstm, hidden = self.recurrent(embed, hidden)
        output_lstm = self.final_dropout(output_lstm)
        # the output is L x 1 x H
        # the last entry of the output is the output 
        # of the network (what we're after) - 1 x H
        output = output_lstm[-1]
        # do the scoring - 1 x V
        scores = self.scoring(output)
        max_score, current_word = torch.max(scores, dim=1)
        # use the most likely word for generated word list
        generated_words.append(current_word)
        # if we predict a sequence, keep going from here
        if n_words > 1:
            for i in range(n_words-1):
                embed = self.embedding(current_word)
                embed = self.emb_dropout(embed)
                # current dimension is 1 x E - turn into 1 x 1 x E
                embed = embed.unsqueeze(0)
                # after LSTM: 1 x 1 x H
                output_lstm, hidden = self.recurrent(embed, hidden)
                # get the first (and only output) - 1 x H
                output = output_lstm[0]
                # get the scores - 1 x H to 1 x V
                scores = self.scoring(output)
                _, current_word = torch.max(scores, dim=1)
                # append this word
                generated_words.append(current_word)
        # turn word list into tensor and return
        generated = torch.stack(generated_words, dim=1)
        return generated

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.model = self.model.cuda()
        
        # optimizer and criterion for this trainer
        # self.optimizer = torch.optim.ASGD(model.parameters(), lr=1e-2, weight_decay=1e-3, t0=1e6)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        # set the model to train mode
        self.model.train()
        epoch_loss = 0
        # run one epoch
        for batch_num, (inputs, targets) in enumerate(self.loader):
            print('\rBatch {:03}/{:03}'.format(batch_num+1, len(self.loader)), end='')
            epoch_loss += self.train_batch(inputs, targets)
        # compute and add the epoch loss
        epoch_loss = epoch_loss / len(self.loader)
        self.train_losses.append(epoch_loss)
        # logging
        self.epochs += 1
        print('\r[TRAIN] Epoch {:03}/{:03} Loss {:7.4f} Perpl {:7.4f}'.format(
            self.epochs, self.max_epochs, epoch_loss, np.exp(epoch_loss)
        ))

    def train_batch(self, inputs, targets):
        if self.cuda:
            inputs = inputs.cuda()
        outputs = self.model(inputs)
        # flatten the output and target for the loss function
        outputs = outputs.view(-1, outputs.size(2))
        targets = targets.view(-1).type(torch.LongTensor)
        if self.cuda:
            targets = targets.cuda()
        # loss of the flattened outputs
        loss = self.criterion(outputs, targets)
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return batch loss
        return loss.item()
    
    def test(self):
        self.model.eval()
        # get predictions
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) 
        self.predictions.append(predictions)
        nll = test_prediction(predictions, fixtures_pred['out'])
        
        # predictions for 20 words
        generated_logits = TestLanguageModel.generation(fixtures_gen, 20, self.model) 
        # predictions for 20 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 20, self.model)

        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        # get predictions
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) 
        self.predictions_test.append(predictions_test)

        # logging
        print('\r[VAL]   Epoch {:03}/{:03} NLL {:7.4f}'.format(
            self.epochs, self.max_epochs, nll
        ))

        return nll

    def save(self):
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()}, model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(str(self.generated[-1].encode('utf-8')))
        with open(os.path.join('experiments', self.run_id, 'generated-test-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(str(self.generated_test[-1].encode('utf-8')))
        with open(os.path.join('experiments', self.run_id, 'train_losses.txt'), 'w') as fw:
            for tloss in self.train_losses:
                fw.write(str(tloss))
        with open(os.path.join('experiments', self.run_id, 'val_losses.txt'), 'w') as fw:
            for vloss in self.val_losses:
                fw.write(str(vloss))

class TestLanguageModel:
    @staticmethod
    def prediction(inp, model):
        """
            :param inp: (batch size, length)
            :return: a np.ndarray of logits
        """
        # transpose from B x L to L x B
        inp = inp.transpose()
        inp = torch.Tensor(inp).type(torch.LongTensor)
        if torch.cuda.is_available():
            model = model.cuda()
            inp = inp.cuda()
        output = model.forward(inp)
        # output is L x B x V, turn into B x V
        # by taking the last element (next word)
        # output = output.view(inp.size(1), VOCAB_SIZE)
        output = output[-1]
        return output.detach().cpu().numpy()
        
    @staticmethod
    def generation(inp, forward, model):
        """
            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
        inp = inp.T
        inp = torch.Tensor(inp).type(torch.LongTensor)
        if torch.cuda.is_available():
            model = model.cuda()
            inp = inp.cuda()
        generated_words = model.generate(inp, forward)
        return generated_words

# Generating a `run_id` and saving all files to the new directory.
run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)

import config as cg

# language model
model = LanguageModel(cg.vocab_size, cg.embed_size, cg.hidden_size, cg.n_layers)

# dataset and data loader
dataset = FixedSequenceDataset(data, cg.seq_length)
loader = DataLoader(dataset, shuffle=True, batch_size=cg.batch_size, collate_fn=collate)

# trainer
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=cg.n_epochs, run_id=run_id)

# initialize best NLL to large value
best_nll = 1e30

# train the model
for epoch in range(cg.n_epochs):
    trainer.train()
    nll = trainer.test()
    # save model whenever we improve NLL
    if nll < best_nll:
        best_nll = nll
        print("\rSaving data for epoch {:} with NLL {:}".format(epoch, best_nll))
        trainer.save()

# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation NLL')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

# see generated output
print(trainer.generated[-1])