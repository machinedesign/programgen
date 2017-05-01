import pytest
from . import minipython
from . import formula

modules = [
    minipython,
    formula
]


@pytest.mark.parametrize("mod", modules)
def test_gen_program(mod):
    for _ in range(20):
        print(mod.gen_code())

@pytest.mark.parametrize("mod", modules)
def test_gen_examples(mod):
    for _ in range(10):
        code = mod.gen_code()
        data = mod.gen_examples(code, 10)

@pytest.mark.parametrize("mod", modules)
def test_exec_code(mod):
    for _ in range(10):
        code = mod.gen_code()
        for _ in range(10):
            inp = mod.gen_input()
            out = mod.exec_code(code, input=inp)
            print(code, inp, out)
