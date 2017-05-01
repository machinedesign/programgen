from collections import deque
from collections import namedtuple
import numpy as np

from lark import Lark
from lark import InlineTransformer

from operator import add, sub, mul, truediv as div, neg

from .common import Program
from .utils import flatten

safe_div = lambda x,y: (x / y) if y != 0 else 0
ADD = 'add'
SUB = 'sub'
MUL = 'mul'
DIV = 'div'
NEG = 'neg'
OPS = (ADD, SUB, MUL, DIV)
UNARY_OPS = (NEG,)
op_func = {
    ADD: add,
    MUL: mul,
    SUB: sub,
    DIV: safe_div,
    NEG: neg,
}


def gen_code(random_state=None):
    rng = np.random.RandomState(random_state)
    tree = _gen_formula_tree(rng)
    return _tree_to_str(tree)


def gen_examples(code, nb_examples=10):
    return _gen_examples(code, nb_examples=nb_examples, symbols=('x',), gen_value=_gen_value_from)


def exec_code(s, input):
    mem = _exec_code(s, input, symbols=('x',), gen_value=_gen_value_from)
    out = mem[-1][0]
    return [out]


def _exec_code(s, input, symbols=None, gen_value=None):
    if symbols is None:
        symbols = ('x',)
    if gen_value is None:
        gen_value = _gen_value_from
    expr = s
    for i, s in enumerate(symbols):
        val = input[i]
        expr = expr.replace(s, str(val))
    opcode = _get_formula_opcode(expr)
    mem = _exec_opcode(opcode)
    return mem


def gen_input(random_state=None):
    return [_gen_value()]


def _gen_examples(code, nb_examples=10, symbols=None, gen_value=None):
    if symbols is None:
        symbols = ('x',)
    if gen_value is None:
        gen_value = _gen_value_from
    mems = []
    vals = []
    inps = []
    outs = []
    for _ in range(nb_examples):
        v = []
        inp = [gen_value(s) for s in symbols]
        inps.append(inp)
        mem = _exec_code(code, inp, symbols=symbols, gen_value=gen_value)
        out = mem[-1][0]
        outs.append(out)
        v.append(out)
        mems.append(mem)
        vals.append(v)
    return Program(code=code, vals=vals, mems=mems, inps=inps, outs=outs)


def _gen_formula_tree(rng, depth=0):
    if depth >= 3:
        action = rng.choice(('val', 'symbol'))
    elif depth <= 2:
        action = 'op'
    else:
        action = rng.choice(('op', 'val', 'symbol', 'unary_op'))
    if action == 'op':
        op = rng.choice(('+', '*'))
        return (op, _gen_formula_tree(rng, depth=depth+1), _gen_formula_tree(rng, depth=depth+1))
    elif action == 'unary_op':
        op = '-'
        return (op, _gen_formula_tree(rng, depth=depth + 1))
    elif action == 'symbol':
        return rng.choice(('x',))
    elif action == 'val':
        return _gen_value()


def _gen_value_from(s):
    return _gen_value()


def _gen_value():
    return np.random.randint(1, 10)


def _tree_to_str(tree):
    if type(tree) == tuple:
        if len(tree) == 3:
            op, left, right = tree
            left = _tree_to_str(left)
            right = _tree_to_str(right)
            if op in ('+', '*'):
                return str(left) + op + str(right)
            else:
                return '(' + str(left) + op + str(right) + ')'
        elif len(tree) == 2:
            op, val = tree
            return op + str(val)
    else:
        return tree


def _get_formula_opcode(f):
    tree = _calc_tree(f)
    opcode = []
    _get_formula_opcode_from_tree(tree, code=opcode)
    return opcode


def _get_formula_opcode_from_tree(t, code=None):
    C = t.children
    if len(C) == 2:
        c1 = _get_formula_opcode_from_tree(C[0], code=code)
        c2 = _get_formula_opcode_from_tree(C[1], code=code)
        if c1:code.append(_push(c1))
        if c2:code.append(_push(c2))
        code.append(_op(t.data))
        return None
    elif len(C) == 1:
        return float(C[0].value)
    else:
        raise ValueError('not expected')


def _exec_opcode(opcode):
    mem = Mem()
    for optype, vals in opcode:
        if optype == 'PUSH':
            mem.append(vals[0])
        elif optype in OPS:
            v2 = mem.pop()
            v1 = mem.pop()
            mem.append(_apply_op(optype, v1, v2))
        elif optype in BINARY_OPS:
            v = mem.pop()
            mem.append(_apply_op(optype, v))
    return mem.evol


class Mem(deque):
    
    def __init__(self):
        super().__init__()
        self.evol = []

    def append(self, x):
        super().append(x)
        self.evol.append(tuple(self))

    def pop(self):
        out = super().pop()
        self.evol.append(tuple(self))
        return out


def _apply_op(t, a, b=None):
    if b is not None:
        return op_func[t](a, b)
    else:
        return op_func[t](a)


def _op(x):
    return (x, [])


def _push(x):
    return ('PUSH', [x])


parser = Lark(
    '''
    ?sum: product
         | sum "+" product   -> add
         | sum "-" product   -> sub

     ?product: item
         | product "*" item  -> mul
         | product "/" item  -> div

     ?item: NUMBER           -> number
          | "-" item         -> neg
          | "(" sum ")"

     %import common.NUMBER
     %import common.WS
     %ignore WS
''', start='sum')


class CalculateTree(InlineTransformer):
    from operator import add, sub, mul, truediv as div, neg
    number = float


def _calc_tree(expr):
    return parser.parse(expr)
