import torch
from time import time


__all__ = ['timer']


def timer(f, *args, text='', text_size=64, **kwargs):
    if isinstance(text, str) and len(text) > 0:
        text = text
    elif hasattr(f, '__name__'):
        text = f.__name__
    elif hasattr(f, '__class__'):
        text = f.__class__.__name__
    else:
        text = ''
    torch.cuda.synchronize()
    start = time()
    out = f(*args, **kwargs)
    torch.cuda.synchronize()
    padding = '.' * (text_size - len(text))
    print(f'{text}{padding}: {time() - start:0.3f}s')
    return out
