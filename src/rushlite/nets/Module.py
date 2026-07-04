from abc import ABC, abstractmethod
from typing import Any, Generator, Iterator

from rushlite._C import Variable


class _ParamsView:
    def __init__(self, module: "Module") -> None:
        self._module = module

    @staticmethod
    def _split(path: str) -> tuple[list[str], str]:
        parts = path.split(".")
        return parts[:-1], parts[-1]

    def _resolve_owner(self, path: str, create: bool = False) -> tuple["Module", str]:
        parents, leaf = self._split(path)
        owner = self._module
        for part in parents:
            val = owner._params_dict.get(part)
            if not isinstance(val, Module):
                raise KeyError(f"No submodule '{part}' along path '{path}'")
            owner = val
        return owner, leaf

    def __getitem__(self, path: str) -> Variable:
        owner, leaf = self._resolve_owner(path)
        val = owner._params_dict.get(leaf)
        if not isinstance(val, Variable):
            raise KeyError(f"No parameter named '{path}'")
        return val

    def __setitem__(self, path: str, value: Variable) -> None:
        owner, leaf = self._resolve_owner(path)
        if not value.requires_grad:
            value = Variable(value.data, requires_grad=True)
        owner._params_dict[leaf] = value

    def __iter__(self) -> Iterator[str]:
        for name, _ in self._module.named_parameters():
            yield name

    def keys(self) -> Iterator[str]:
        return iter(self)


class Module(ABC):
    def __init__(self) -> None:
        self._params_dict = {}

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name == "_params_dict":
            super().__setattr__(name, value)
            return

        params_dict = self.__dict__.get("_params_dict")
        if params_dict is None:
            raise AttributeError(
                "call super().__init__() before assigning/accessing parameters"
            )

        if isinstance(value, (Module, Variable)):
            params_dict[name] = value
            return

        params_dict.pop(name, None)
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        params_dict = self.__dict__.get("_params_dict")
        if params_dict is None:
            raise AttributeError(
                "call super().__init__() before assigning/accessing parameters"
            )
        try:
            return params_dict[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __delattr__(self, name: str) -> None:
        params_dict = self.__dict__.get("_params_dict")
        if params_dict is not None and name in params_dict:
            del params_dict[name]
            return
        super().__delattr__(name)

    @property
    def params(self) -> _ParamsView:
        return _ParamsView(self)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

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

    def zero_grad(self) -> None:  # TODO
        pass

    def to(self) -> None:  # TODO
        pass

    def __repr__(self) -> str:  # TODO
        return super().__repr__()
