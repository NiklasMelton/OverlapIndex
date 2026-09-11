"""Fail-loud import shim for the archived smallNORB discovery runner."""

DATASET_ID = "smallnorb"


def load_dataset(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("the archived smallNORB discovery source is unavailable")

