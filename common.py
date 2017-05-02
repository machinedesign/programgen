from collections import namedtuple
import hashlib

from joblib import Parallel, delayed
from joblib import Memory

Program = namedtuple('Program', ['code', 'vals', 'mems', 'inps', 'outs'])

mem = Memory(cachedir='.cache')

@mem.cache
def gen_programs(gen_input, gen_code, exec_code, nb_programs=100):
    inputs = [gen_input() for _ in range(64)]
    programs = _gen_programs_pool(gen_code)
    exists = set()
    p = []
    for program in programs:
        try:
            chck = program_checksum(program, inputs, exec_code)
        except Exception:
            chck = None
        if chck in exists or chck is None:
            continue
        else:
            p.append(program)
            print(len(p))
            exists.add(chck)
            if len(p) == nb_programs:
                break
    return p


@mem.cache
def _gen_programs_pool(gen_code):
    programs = [gen_code() for _ in range(100000)]
    return programs

def program_checksum(code, inputs, exec_code):
    m = hashlib.sha256()
    for inp in inputs:
        out = exec_code(code, input=inp)
        out = tuple(out)
        m.update(str(out).encode('utf8'))
    return m.hexdigest()
