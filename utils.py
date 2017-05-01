def padded(lists, pad=0, max_length=None):
    if max_length is None:
        max_length = max(map(len, lists))
    return [list(l) + [pad] * (max_length - len(l)) for l in lists]
