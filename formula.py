from collections import deque
from collections import namedtuple
import numpy as np

from lark import Lark
from lark import InlineTransformer

Formula = namedtuple('Formula', ['expr', 'vals', 'mems'])

ADD = 'add'
MUL = 'mul'
OPS = (ADD, MUL)
op_func = {
    ADD: lambda a, b: a + b,
    MUL: lambda a, b: a * b
}


def gen_formula(random_state=None):
    rng = np.random.RandomState(random_state)
    tree = _gen_formula_tree(rng)
    s = _tree_to_str(tree)
    return s


def _gen_value_from(s):
    return _gen_value()


def _gen_value():
    return np.random.randint(1, 10)


def build_single_formula_data(expr_tpl, nb_examples=10, symbols=['x'], gen_value=_gen_value_from):
    mems = []
    vals = []
    for _ in range(nb_examples):
        v = []
        expr = expr_tpl
        for s in symbols:
            val = gen_value(s)
            expr = expr.replace(s, str(val))
            v.append(val)
        opcode = _get_formula_opcode(expr)
        mem = _exec_opcode(opcode)
        v.append(mem[-1][0])
        mems.append(mem)
        vals.append(v)
    return Formula(expr=expr_tpl, vals=vals, mems=mems)


def _gen_formula_tree(rng, depth=0):
    if depth >= 3:
        action = rng.choice(('val', 'symbol'))
    elif depth <= 2:
        action = 'op'
    else:
        action = rng.choice(('op', 'val', 'symbol'))
    if action == 'op':
        op = rng.choice(('+', '*'))
        return (op, _gen_formula_tree(rng, depth=depth+1), _gen_formula_tree(rng, depth=depth+1))
    elif action == 'symbol':
        return rng.choice(('x',))
    elif action == 'val':
        return _gen_value()


def _tree_to_str(tree):
    if type(tree) == tuple:
        op, left, right = tree
        left = _tree_to_str(left)
        right = _tree_to_str(right)
        return '(' + str(left) + op + str(right) + ')'
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
            v1 = mem.pop()
            v2 = mem.pop()
            mem.append(_apply_op(optype, v1, v2))
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


def _apply_op(t, a, b):
    return op_func[t](a, b)


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
