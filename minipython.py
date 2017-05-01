from collections import Counter
from collections import OrderedDict
from collections import namedtuple

import numpy as np

Program = namedtuple('Program', ['code', 'vals', 'mems', 'inps', 'outs'])

class dotdict(OrderedDict):

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return []


def gen_code(random_state=None):
    while True:
        code = _gen_code()
        try:
            cnt = Counter(code)                
            for k, v in cnt.items():
                if k != 'vars.a' and type(k) != int and k.startswith('vars.'):
                    assert v >= 2
            for _ in range(10):
                _exec_code(code, a=gen_input())
        except Exception as ex:
            print(ex)
            print(to_str(code))
            continue
        else:
            break
    return code


def _gen_code(random_state=None):
    rng = np.random.RandomState(random_state)
    nb_steps = 3
    names = 'abcdefghijklmnopqrstuvwxyz'
    funcs = ['map_', 'filter_', 'count', 'head', 'min_', 'max_', 'sort', 'reverse', 'sum_', 'tail']
    arith = ['plus', 'minus', 'div']
    predicate = ['pos', 'neg']
    ops ={
        'map_': arith,
        'filter_': predicate,
        'count': predicate
    } 
    nb_inputs = 1
    toks = []
    for i in range(nb_inputs, nb_inputs + nb_steps):
        name = names[i]
        func = rng.choice(funcs)
        if func in ('map_', 'filter_', 'count'):
            op = rng.choice(ops[func])
            v = ['vars.', names[rng.randint(0, i)]]
            if op in arith:
                n = str(rng.randint(1, 5))
                op = [op, '(', n, ')']
            else:
                op = [op]
            val  = [func, '('] + op +  [','] + v + [')']
        elif func in ('take', 'drop', 'access'):
            n = str(rng.randint(1, 5))
            v = names[rng.randint(0, i)]
            val = [func, '(', n, ',', 'vars.', v, ')']
        else:
            v = names[rng.randint(0, i)]
            val = [func, '(', 'vars.', v, ')']
        stmt = ['vars.', name, ' = '] + val + ['\n']
        toks.extend(stmt)
    return toks

map_ = lambda x,y:list(map(x, y))
filter_ = lambda x,y:list(filter(x, y))
sort = sorted

plus = lambda n:(lambda x:x+n)
minus = lambda n:(lambda x:x-n)
mul = lambda n:(lambda x:x * n)
div = lambda n:(lambda x:x // n)
pow = lambda n:(lambda x:x ** n)

pos = lambda x:x > 0
neg = lambda x:x < 0
head = lambda x:[x[0]]
tail = lambda x:[x[-1]]
min_ = lambda x:[min(x)]
max_ = lambda x:[max(x)]
sum_ = lambda x:[sum(x)]
reverse = lambda x:list(reversed(x))
take = lambda n, x:x[:n]
drop = lambda n, x:x[n:]
access = lambda n, x: [x[n]]
count = lambda x, y: [sum(map(x, y))]

def exec_code(s, input):
    vars = _exec_code(s, a=input)
    out = vars[list(vars.keys())[-1]]
    return out


def _exec_code(s, **kwargs):
    s = to_str(s)
    vars = dotdict()
    vars.update(kwargs)
    exec(s)
    return vars


def to_str(sent, sep=''):
    sent = [s for s in sent if s not in (0, 1, 2, 3)]
    return sep.join(sent)


def gen_examples(code, nb_examples=10):
    vals = []
    mems = []
    inps = []
    outs = []
    for _ in range(nb_examples):
        inp = gen_input()
        vars = _exec_code(code, a=inp)
        out = vars[list(vars.keys())[-1]]
        vals.append(inp + out)
        #mems.append([[ve for v in vars.values() for ve in v]])
        mems.append(list(vars.values()))
        #mems.append([ [f for e in list(vars.values())[0:i] for f in e] for i in range(len(vars))])
        inps.append(inp)
        outs.append(out)
    return Program(code=code, vals=vals, mems=mems, inps=inps, outs=outs)


def gen_input(random_state=None):
    rng = np.random.RandomState(random_state)
    length = rng.randint(1, 5)
    return [rng.randint(1, 10) for _ in range(length)]


if __name__ == '__main__':
    code = (gen_code())
    print(code)
    vars = _exec_code(code, a=[1, 2, 3, 4])
    print(vars)
