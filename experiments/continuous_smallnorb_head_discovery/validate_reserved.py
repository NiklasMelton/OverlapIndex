"""Fail-loud import shim for the archived smallNORB reserved validator."""


def load_dataset(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("the archived smallNORB reserved source is unavailable")
