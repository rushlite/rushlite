from abc import ABC, abstractmethod
from typing import Any, Generator, Iterator

from rushlite._C import Variable


class Module(ABC):
    """Base class for neural network modules.

    Parameters (Variable) and submodules (Module) assigned as attributes are
    stored ONLY in ``_params_dict`` -- never in the instance ``__dict__`` --
    so ``self.w`` and ``parameters()`` always see the same object and a
    functional update through ``params`` reaches ``forward`` immediately.
    Consequently ``vars(module)`` does not list parameters, but ``hasattr``
    / ``getattr`` still resolve them via ``__getattr__``.
    """

    def __init__(self) -> None:
        self._params_dict = {}

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __setattr__(self, name: str, value: Any, /) -> None:
        params = self.__dict__.get("_params_dict")
        if isinstance(value, (Module, Variable)):
            if params is None:
                raise AttributeError(
                    f"cannot assign parameter {name!r}: "
                    "call super().__init__() before assigning/accessing parameters"
                )
            params[name] = value
        else:
            if params is not None:
                # a plain attribute shadows any parameter of the same name;
                # evict it so parameters() does not yield a stale Variable
                params.pop(name, None)
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        params = self.__dict__.get("_params_dict")
        if params is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}: "
                "call super().__init__() before assigning/accessing parameters"
            )
        if name in params:
            return params[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __delattr__(self, name: str, /) -> None:
        params = self.__dict__.get("_params_dict")
        if params is not None and name in params:
            del params[name]
        else:
            super().__delattr__(name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def params(self) -> "_ParamsView":
        """Mapping view over the module tree's parameters, keyed by the
        dotted paths of ``named_parameters()``. Supports functional updates:

            model.params["layer1.w"] = rushlite.Variable(
                p.data - lr * p.grad, requires_grad=True
            )

        Assignment stores the Variable as-is: its ``requires_grad`` flag is
        preserved, so a value built under ``no_grad()`` comes out frozen (see
        ``_ParamsView``). Note that ``incr_grad`` rebinds the grad tensor, so
        a ``p.grad`` handle is a snapshot: re-read it after each ``backward()``
        rather than caching it across steps.
        """
        return _ParamsView(self)

    def parameters(self) -> Generator[Variable, None, None]:
        for val in self._params_dict.values():
            if isinstance(val, Variable):
                yield val
            elif isinstance(val, Module):
                yield from val.parameters()
            else:
                raise Exception()

    def named_parameters(
        self, _path: list[str] | None = None
    ) -> Generator[tuple[str, Variable], None, None]:
        _path = _path or []
        for key, val in self._params_dict.items():
            if isinstance(val, Variable):
                yield (".".join(map(str, (_path + [key]))), val)
            elif isinstance(val, Module):
                yield from val.named_parameters(_path=_path + [key])
            else:
                raise Exception()

    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.requires_grad:
                p.zero_grad()

    def to(self) -> None:  # TODO
        pass

    def __repr__(self) -> str:  # TODO
        return super().__repr__()


class _ParamsView:
    """Dict-like view over a Module tree's parameters.

    Keys are the dotted paths yielded by ``named_parameters()`` (e.g.
    "layer1.w"). ``__setitem__`` walks to the owning module and assigns
    there, preserving the value's ``requires_grad`` flag: an update
    expression evaluated under ``no_grad()`` comes out with
    ``requires_grad=False`` and is stored that way (the param is frozen).
    To keep a parameter trainable, rewrap explicitly:
    ``Variable(v.data, requires_grad=True)`` (cheap -- the tensor is a
    shared handle).
    """

    def __init__(self, module: Module) -> None:
        self._module = module

    def _owner(self, key: str) -> tuple[Module, str]:
        *path, leaf = key.split(".")
        module = self._module
        for part in path:
            sub = module._params_dict.get(part)
            if not isinstance(sub, Module):
                raise KeyError(key)
            module = sub
        return module, leaf

    def __getitem__(self, key: str) -> Variable:
        module, leaf = self._owner(key)
        param = module._params_dict.get(leaf)
        if not isinstance(param, Variable):
            raise KeyError(key)
        return param

    def __setitem__(self, key: str, value: Variable) -> None:
        if not isinstance(value, Variable):
            raise TypeError(
                f"can only assign a Variable to parameter {key!r}, "
                f"got {type(value).__name__}"
            )
        module, leaf = self._owner(key)
        setattr(module, leaf, value)

    def __iter__(self) -> Iterator[str]:
        for name, _ in self._module.named_parameters():
            yield name

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        try:
            self[key]  # type: ignore[index]
        except (KeyError, AttributeError, TypeError):
            return False
        return True

    def keys(self) -> Iterator[str]:
        yield from self

    def items(self) -> Generator[tuple[str, Variable], None, None]:
        yield from self._module.named_parameters()

    def __repr__(self) -> str:
        return f"params({list(self)})"
