

from __future__ import annotations

from typing import Any, Mapping

class Dict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def update(self, *args, **kwargs) -> None:

        def _wrap(v):
            if isinstance(v, Dict):
                return v
            if isinstance(v, Mapping):
                return Dict({kk: _wrap(vv) for kk, vv in v.items()})
            if isinstance(v, list):
                return [_wrap(x) for x in v]
            if isinstance(v, tuple):
                return tuple(_wrap(x) for x in v)
            return v

        if args:
            if len(args) != 1:
                raise TypeError("update expected at most 1 positional argument")
            other = args[0]
            if isinstance(other, Mapping):
                for k, v in other.items():
                    super().__setitem__(k, _wrap(v))
            else:
                for k, v in dict(other).items():
                    super().__setitem__(k, _wrap(v))
        for k, v in kwargs.items():
            super().__setitem__(k, _wrap(v))

