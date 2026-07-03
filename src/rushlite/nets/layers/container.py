from typing import Any, Sequence as PySequence

from ..Module import Module


class Sequential(Module):
    def __init__(self, layers: "PySequence[Module]") -> None:
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            self._params_dict[str(i)] = layer

    def forward(self, *args: Any) -> Any:
        if not self.layers:
            raise ValueError("Must have at least one layer")
        out = self.layers[0](*args)
        for layer in self.layers[1:]:
            out = layer(out)
        return out
