import time
from collections import defaultdict
from clize import run

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import logging

from machinedesign.transformers import DocumentVectorizer

from . import formula
from . import minipython
from .common import program_checksum
from .common import gen_programs
from .utils import padded
from .utils import to_str

class Simple(nn.Module):

    def __init__(self, nb_examples=10, nb_features=2, emb_size=50, mem_size=50, mem_len=10, hidden_size=32, vocab_size=10):
        super().__init__()
        self.mem_len = mem_len
        self.mem_size = mem_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.repr_features = nn.Sequential(
            nn.Linear(nb_features, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True)
        )
        self.pred_mem_val = nn.Linear(hidden_size, mem_len * mem_size)
        self.mem_repr = nn.Linear(mem_len * mem_size, hidden_size)
        self.text_rnn = nn.LSTM(emb_size + hidden_size, hidden_size, batch_first=True)
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.text_out =  nn.Linear(hidden_size, vocab_size)
 
    
    def forward(self, X, M, T):
        X = self.repr_features(X)
        T = self.emb(T)
        om_val = self.pred_mem_val(X)
        m = self.mem_repr(om_val)
        m = m.mean(0).view(1, 1, -1)
        m = m.repeat(1, T.size(1), 1)
        t = torch.cat((T, m), 2)
        ot, (ht, ct) = self.text_rnn(t)
        ot = ot.view(-1, ot.size(2))
        ot = self.text_out(ot)
        om_val = om_val.view(om_val.size(0), self.mem_len, self.mem_size).view(-1, self.mem_size)
        return om_val, ot
    
    def generate(self, X, mem_len=13, text_len=20, cuda=False, greedy=False):
        X = self.repr_features(X)
        m = self.pred_mem_val(X)
        m = self.mem_repr(m)
        m = m.mean(0).view(1, 1, -1)

        T = torch.ones(1, 1).long()
        T = Variable(T)
        if cuda:
            T = T.cuda()
        ht, ct = torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)
        ht = Variable(ht)
        ct = Variable(ct)
        if cuda:
            ht = ht.cuda()
            ct = ct.cuda()
        t = []
        for _ in range(text_len):
            T_emb = self.emb(T.detach())
            x = torch.cat((T_emb, m), 2)
            ot, (ht, ct) = self.text_rnn(x, (ht, ct))
            ot = ot.view(-1, self.hidden_size)
            ot = nn.Softmax()(self.text_out(ot))
            ot = ot.view(-1)
            if greedy:
                ot = ot.max(0)[1]
            else:
                ot = torch.multinomial(ot)
            T.data[:, :] = ot.data[0]
            t.append(ot.data[0])
            if ot.data[0] == 2:
                break
        return t
   


class RNN(nn.Module):

    def __init__(self, nb_examples=10, nb_features=2, emb_size=50, mem_size=30, mem_len=10, hidden_size=32, vocab_size=10):
        """
        Parameters
        ==========

        nb_examples : total nb examples per program  to generate
        nb_features : max nb of features to represent the inputs and outputs of a program.
                      for instance if the program has 1 input and 1 output, then it should be 2.
        emb_size    : embedding size for the code of the program
        mem_size    : max memory size
        hidden_size : size of LSTM hidden state
        vocab_size  : size of vocabulary for the program code
        """
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.mem_size = mem_size

        # inputs/outputs encoder
        self.enc = nn.Linear(nb_features, emb_size)
        # initialization of LSTM that predicts memory states
        # using the representation of inputs and outputs
        self.init_h = nn.Linear(emb_size, hidden_size)
        self.init_c = nn.Linear(emb_size, hidden_size)
        # encoder for LSTM hidden state of the LSTM that predicts the program code
        # used for attention
        self.text_repr = nn.Linear(hidden_size, hidden_size)
        # encoder for LSTM hidden state of the LSTM that predicts the memory states
        # used for attention
        self.mem_repr = nn.Linear(hidden_size, hidden_size)
        # used for attention to get the coefcient of each pair
        self.repr = nn.Linear(hidden_size, 1)
        # LSTM that predicts the memory states
        self.mem_rnn = nn.LSTM(mem_size, hidden_size, batch_first=True)
        # LSTM that predicts the program code
        self.text_rnn = nn.LSTM(emb_size, hidden_size, batch_first=True)
        # predicts memory states using LSTM hidden state
        self.mem_out = nn.Sequential(nn.Linear(hidden_size, mem_size))
        # predicts probability of next character of program code given LSTM hidden state
        self.text_out =  nn.Linear(hidden_size * 2, vocab_size)
        # embedding of program code
        self.emb = nn.Embedding(vocab_size, emb_size)

    def forward(self, X, M, T):
        T = self.emb(T)
        X = self.enc(X)
        init_h = self.init_h(X)
        init_c = self.init_c(X)
        init_h = init_h.view(1, init_h.size(0), init_h.size(1))
        init_c = init_c.view(1, init_c.size(0), init_c.size(1))
        # get LSTM hidden states for memory state predictor
        om, (hm, cm) = self.mem_rnn(M, (init_h, init_c))
        # get LSTM hidden states for for program code predictor
        ot, (ht, ct) = self.text_rnn(T)
        # Apply the attention mechanism to align memory LSTM
        # and code LSTM
        mt = self.attention(om, ot)
        # Get the final representation of each timestep of the code predictor
        ot = self.trepr(ot, mt)
        ot = ot.contiguous()
        ot = ot.view(-1, self.hidden_size * 2)
        # -- Predict the next character of program code
        ot = self.text_out(ot)

        # Get the prediction of memory states and controller state
        om = om.contiguous()
        om = om.view(-1, self.hidden_size)
        # -- Predict the next memory state
        om_val = self.mem_out(om)
        return om_val, ot
    
    def attention(self, om, ot):
        return self._attention_fast(om, ot)
        #return self._attention_baseline(om, ot)

    def _attention_baseline(self, om, ot):
        return om[:, :].mean(0).mean(1).repeat(1, ot.size(1), 1)

    def _attention_fast(self, om, ot):
        t = self.text_repr(ot.view(-1, ot.size(2)))
        t = t.view(ot.size(0), ot.size(1), t.size(1))
        
        m = self.mem_repr(om.contiguous().view(-1, om.size(2)))
        m = m.view(om.size(0), om.size(1), m.size(1))
        m = m.mean(0)
        
        nb_t = t.size(1)
        nb_m = m.size(1)

        t = t.repeat(nb_m, 1, 1)
        t = t.transpose(0, 1)
        t = t.contiguous()
        m = m.repeat(nb_t, 1, 1)

        u = nn.Tanh()(m + t)
        u = u.view(-1, u.size(2))
        u = self.repr(u)
        u = u.view(nb_t, nb_m)
        u = nn.Softmax()(u)
        u = u.view(nb_t, nb_m, 1)
        u = u.repeat(1, 1, m.size(2))
        mt = (u * m).sum(1)
        mt = mt.transpose(0, 1)
        mt = mt.contiguous()
        return mt
    
    def _attention_slow(self, om, ot):
        mts = []
        # For each pair of timesteps from ot (indexed by i) and om (indexed by j)
        # compute a scalar coeficient based on how much they match, then
        # compute a weighted sum of om weighted by the coeficients
        for i in range(ot.size(1)):
            p = []
            mt = []
            for j in range(om.size(1)):
                t = ot[:, i, :]
                m = om[:, j, :]
                # get memory state representation for each timestep j
                m = self.mem_repr(m)
                m = m.mean(0) # pooling over the examples to make it indep on nb of examples
                mt.append(m)
                # get code representation for each timestep i
                t = self.text_repr(t)
                # computing how much i and j match and get the coeficient
                u = self.repr(nn.Tanh()(m + t))
                p.append(u.view(-1)) # p has the list of real-valued coeficients
            # make the coeficients a probabiliy distribution using softmax
            p = torch.cat(p, 0)
            p = nn.Softmax()(p.view(1, -1))
            # compute a weighted sum of the memory representations
            mt = torch.cat(mt, 0)
            p = p.view(-1, 1).repeat(p.size(0), mt.size(1))
            mt = (mt * p).sum(0)
            mts.append(mt)
        # return the weighted sum of memory representations at each timestep i
        mt = torch.cat(mts, 0)
        mt = mt.view(1, mt.size(0), mt.size(1))
        # shape of mt : (1, max_trace_length, mem_size)
        return mt

    def generate(self, X, mem_len=13, text_len=20, cuda=False, greedy=False):
        X = self.enc(X)
        init_h = self.init_h(X)
        init_c = self.init_c(X)
        init_h = init_h.view(1, init_h.size(0), init_h.size(1))
        init_c = init_c.view(1, init_c.size(0), init_c.size(1))
        M = torch.zeros(X.size(0), 1, self.mem_size)
        M = Variable(M)
        if cuda:
            M = M.cuda()

        hm, cm = init_h, init_c
        oms = []
        # Predict next memory states and feed back as input
        # the stopping or mem_len is very important to get right when using attention
        # otherwise it generates garbage
        for _ in range(mem_len):
            om, (hm, cm) = self.mem_rnn(M, (hm, cm))
            oms.append(om)
            om = om.contiguous()
            om = om.view(-1, self.hidden_size)
            om_val = self.mem_out(om)
            M.data.copy_(om_val.data.view(om_val.size(0), 1, om_val.size(1)))
        om = torch.cat(oms, 1) 
        # At this point, we have a list of memory states for each example
        # Now, predict the code character given the predicted list of memory states
        # and current code character

        T = torch.ones(1, 1).long()
        T = Variable(T)
        if cuda:
            T = T.cuda()

        ht, ct = torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)
        ht = Variable(ht)
        ct = Variable(ct)

        if cuda:
            ht = ht.cuda()
            ct = ct.cuda()
        
        t = []
        for _ in range(text_len):
            
            T_emb = self.emb(T.detach())
            ot, (ht, ct) = self.text_rnn(T_emb, (ht, ct))
            mt = self.attention(om, ot)
            ot = self.trepr(ot, mt)
            ot = ot.contiguous()
            ot = ot.view(-1, self.hidden_size * 2)
            ot = nn.Softmax()(self.text_out(ot))
            ot = ot.view(-1)
            if greedy:
                ot = ot.max(0)[1]
            else:
                ot = torch.multinomial(ot)
            T.data[:, :] = ot.data[0]
            t.append(ot.data[0])
            if ot.data[0] == 2:
                break
        return t
   
    def trepr(self, ot, mt):
        return torch.cat((mt, mt), 2)


def train(*, nb_epochs=100, nb_programs=10, nb_examples=64, 
          nb_features=None, emb_size=100, mem_size=None, mem_len=None,
          hidden_size=128, lr=1e-3, cuda=False, test_ratio=0.1, 
          mod='minipython',
          outf='.',
          log_level='DEBUG',
          nb_trials=10):
    """
    Parameters
    ==========


    nb_programs : total nb of programs to generate
    lr          : learning rate
    cuda        : enable/disable CUDA

    For the others, check the doc of RNN
    """
    np.random.seed(43)
    
    stream = logging.StreamHandler()
    stream.setLevel(log_level)

    hndl = logging.FileHandler('{}/train.log'.format(outf), mode='w')
    hndl.setLevel(logging.DEBUG)
    hndl_test = logging.FileHandler('{}/test.log'.format(outf), mode='w')
    hndl_test.setLevel(logging.DEBUG)

    log = logging.getLogger('train')
    log.setLevel(logging.DEBUG)
    log.addHandler(stream)
    log.addHandler(hndl)
    
    log_test = logging.getLogger('test')
    log_test.setLevel(logging.DEBUG)
    log_test.addHandler(stream)
    log_test.addHandler(hndl_test)
    
    mod = {'minipython': minipython, 'formula': formula}[mod]
    gen_program = mod.gen_code
    gen_examples = mod.gen_examples
    exec_code = mod.exec_code
    gen_input = mod.gen_input

    # generate programs and make sure there are no duplicates
    programs = gen_programs(mod.gen_input, mod.gen_code, mod.exec_code, nb_programs=nb_programs)
    max_code_size = max(map(len, programs))
    
    length = max(map(len, programs)) + 2 # we add 2 because of begin and end characters
    doc = DocumentVectorizer(length=length, begin_character=True, end_character=True)
    doc.partial_fit(programs)

    # build dataset
    # each example is a program
    # for each program we have nb_examples set of inputs and their corresponding outputs
    # where the outputs are obtained by executing the program on the inputs
    dataset = []
    
    for program in programs:
        data = gen_examples(program, nb_examples=nb_examples)
        dataset.append(data)
    
    if mem_size is None:
        # if not specified, take the maximum mem length across all traces of programs on all inputs
        mem_size = max(len(n) for e in dataset for m in e.mems for n in m)

    if nb_features is None:
        nb_features = max(len(v) for e in dataset for v in e.vals)

    if mem_len is None:
        mem_len = max(len(m) for e in dataset for m in e.mems)
    
    vocab_size = len(doc.words_)
    rnn = Simple(
        nb_examples=nb_examples, 
        nb_features=nb_features, 
        emb_size=emb_size,
        mem_size=mem_size,
        mem_len=mem_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    if cuda:
        rnn = rnn.cuda()

    optim = torch.optim.Adam(rnn.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
 
    stats = defaultdict(list)
    nb_updates = 0
    avg_loss = 0.
    avg_acc = 0.
    avg_score = 0.
    avg_perfect_score = 0.
    avg_r2_mem = 0.
    gamma = 0.99
    
    nb_train = int((1 - test_ratio) * len(dataset))
    dataset_train = dataset[0:nb_train]
    dataset_test = dataset[nb_train:]
    test_scores = []
    perfect_test_scores = []
    for epoch in range(nb_epochs):
        np.random.shuffle(dataset_train)
        for datapoint in dataset_train:
            t0 = time.time()
            rnn.zero_grad()
            code = datapoint.code
            vals = datapoint.vals
            mems = datapoint.mems
            inps = datapoint.inps
            outs = datapoint.outs

            # transform the code into a list of ints
            expr, = doc.transform([code])
            expr = torch.LongTensor(expr)
            expr = expr.view(1, expr.size(0))

            # pad the memory of each example with zeros to have mem_size size
            # and make the first state of the memory full of zeros
            # mem has shape (nb_examples, max_trace_length, mem_size)
            mems = [padded([[]] + mem, max_length=mem_size) for mem in mems]

            # TODO : it will not work if the examples don't have the same number
            # of steps for the memory states.
            # To make it work, I should do padding here.
            mem = torch.FloatTensor(mems)
            #mem = torch.log(1 + mem)
            # X contains the inputs and outputs contatenated
            vals = [padded([v], max_length=nb_features)[0] for v in vals]
            X = Variable(torch.FloatTensor(vals))
            
            # we give current memory state that we give to LSTM
            M_inp = Variable(mem[:, 0:-1])
            # we ask the LSTM to predict the next memory state
            M_out = Variable(mem[:, 1:].contiguous().view(-1, mem_size))

            # we condition the generated program code on the memory states
            # and on the current character of the program code
            T_inp = Variable(expr[:, 0:-1])
            # we predict the next character of the program code
            T_out = Variable(expr[:, 1:]).view(-1)
            
            if cuda:
                X = X.cuda()
                M_inp = M_inp.cuda()
                M_out = M_out.cuda()
                T_inp = T_inp.cuda()
                T_out = T_out.cuda()

            # om_val is the predicted memory states at each timetep
            # it has shape : (nb_examples * max_trace_length, mem_size)
            om_val, ot = rnn(X, M_inp, T_inp)
            
            loss_mem_val = ((om_val - M_out) ** 2).mean() # mean sqr error for predicting memory states
            loss_code = crit(ot, T_out) # predict the correct next character for program code
            loss =  loss_mem_val + loss_code
            loss.backward()
            nn.utils.clip_grad_norm(rnn.parameters(), 2)
            optim.step()
            
            res = ((om_val.view(-1) - M_out.view(-1))**2).mean()
            tot = M_out.view(-1).var()
            r2_mem_val = 1. - res / tot

            dt = time.time() - t0
            
            if nb_updates % 100 == 0:
                max_score = 0.
                best = None
                for _ in range(nb_trials):
                    t = rnn.generate(X, mem_len=mem_len, text_len=max_code_size, cuda=cuda, greedy=False)
                    t, = doc.inverse_transform([t])
                    score = unit_test(t, inputs=inps, outputs=outs, exec_code=exec_code, print=log.debug)
                    if score > max_score or best is None:
                        max_score = score
                        best = t
                t = best
                log.debug('Max score : {:.3f}'.format(max_score))
                avg_score = avg_score * 0.9 + max_score * 0.1
                avg_perfect_score = avg_perfect_score * 0.9 + (max_score==1.) * 0.1
                log.info('****** Real')
                log.info(to_str(code))
                log.info('****** Generated')
                log.info(to_str(t))
                log.info('****** Unit test')
                total = 0
                correct = 0
                for _ in range(10):
                    inp = gen_input()
                    out = exec_code(code, input=inp)
                    try:
                        pred_out = exec_code(t, input=inp)
                    except Exception as ex:
                        pred_out = None
                    correct += (pred_out == out)
                    total += 1
                    log.debug('Input : {}, Output : {}, Predicted : {}'.format(inp, out, pred_out))
                log.debug('Passed : {}/{}'.format(correct, total))
                torch.save(rnn, '{}/model.pth'.format(outf))
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(outf))
            
            code_acc = acc(ot, T_out)
            stats['loss'].append(loss.data[0])
            stats['loss_mem'].append(loss_mem_val.data[0])
            stats['r2_mem'].append(r2_mem_val.data[0])
            stats['loss_code'].append(loss_code.data[0])
            stats['acc'].append(code_acc.data[0])
            stats['dt'].append(dt)
            
            avg_loss = avg_loss * gamma + stats['loss'][-1] * (1 - gamma)
            avg_acc = avg_acc * gamma + stats['acc'][-1] * (1 - gamma)
            avg_r2_mem = avg_r2_mem * gamma + stats['r2_mem'][-1] * (1 - gamma)
            stats['avg_loss'].append(avg_loss)
            stats['avg_acc'].append(avg_acc)
            stats['avg_score'].append(avg_score)
            stats['avg_perfect_score'].append(avg_perfect_score)

            if nb_updates % 10 == 0:
                s = ['{} : {:.3f}'.format(k, v[-1]) for k, v in stats.items()]
                s = ' '.join(s)
                s = 'epoch {:03d} '.format(epoch) + s
                log.info(s)
            nb_updates += 1
        # Testing after the end of each epoch
        if epoch % 10 == 0:
            t0 = time.time()
            cur_test_scores = []
            for datapoint in dataset_test:
                t0 = time.time()
                code = datapoint.code
                vals = datapoint.vals
                inps = datapoint.inps
                outs = datapoint.outs
                vals = [padded([v], max_length=nb_features)[0] for v in vals]
                X = Variable(torch.FloatTensor(vals))
                if cuda:
                    X = X.cuda()
                max_score = 0.
                best = None
                for _ in range(nb_trials):
                    t = rnn.generate(X, mem_len=mem_len, text_len=max_code_size, cuda=cuda, greedy=False)
                    t, = doc.inverse_transform([t])
                    log_test.debug('Groundtruth : {}'.format(to_str(code)))
                    score = unit_test(t, inputs=inps, outputs=outs, exec_code=exec_code, print=log_test.debug)
                    if score > max_score or best is None:
                        max_score = score
                        best = t
                t = best
                cur_test_scores.append(max_score)

            test_score = np.mean(cur_test_scores)
            test_scores.append(test_score)

            perfect_test_score = (np.array(cur_test_scores) == 1).mean()
            perfect_test_scores.append(perfect_test_score)

            pd.DataFrame({'test_score': test_scores, 'perfect_test_score': perfect_test_scores}).to_csv('{}/stats_test.csv'.format(outf))
            dt = time.time() - t0
            log_test.info('test score : {:.3f} perfect test score : {:.3f} time : {:.3f}'.format(test_score, perfect_test_score, dt))

 
def acc(pred, true):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true).float().mean()
    return acc


def unit_test(code, inputs, outputs, exec_code=minipython.exec_code, print=print):
    correct = 0
    print('Unit test of : {}'.format(to_str(code)))
    for inp, out in zip(inputs, outputs):
        try:
            pred_out = exec_code(code, input=inp)
        except Exception as ex:
            pred_out = None
        print('Predicted : {} Correct : {}'.format(pred_out, out))
        try:
            is_correct = np.all(np.isclose(pred_out, out))
        except Exception:
            is_correct = 0
        correct += is_correct
    print('Passed : {}/{}'.format(correct, len(inputs)))
    return correct / float(len(inputs))

if __name__ == '__main__':
    run(train)
