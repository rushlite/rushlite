## What does this PR do?

<!-- A short description of the change and the motivation for it.
     Link the related issue if there is one, e.g. "Closes #123". -->

## Type of change

<!-- Delete the ones that don't apply. -->

- Bug fix
- New feature (op / layer / API)
- C++/CUDA core (Lamp3)
- Refactor / cleanup
- Docs / CI / build

## How was this tested?

<!-- e.g. "added a graph template", "new GTest case in tensor_tests",
     "ran `uv run pytest tests/` on CPU". If the change touches CUDA code,
     say whether you were able to test on a GPU or are relying on ci-gpu. -->

## Checklist

- [ ] Tests pass locally (`uv run pytest tests/` — CPU is fine)
- [ ] Formatters/linters pass (`black`, `ruff`; `format`/`lint` targets for C++)
- [ ] Added or updated tests where it makes sense
- [ ] If this can't affect the CUDA code, I've cancelled the `ci-gpu` run to save costs
