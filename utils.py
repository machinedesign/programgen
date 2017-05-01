def padded(lists, pad=0, max_length=None):
    if max_length is None:
        max_length = max(map(len, lists))
    return [list(l) + [pad] * (max_length - len(l)) for l in lists]

def flatten(l):
    return [e for ll in l for e in ll]


def to_str(sent, sep=''):
    sent = [s for s in sent if s not in (0, 1, 2, 3)]
    return sep.join(sent)
