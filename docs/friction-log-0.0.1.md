# Friction log — rushlite v0.0.1

A "walk the store" pass over the library as of `5257420`: install from source per the
README, run every README example verbatim, then use the API the way a PyTorch user
would, and finally run the test suite as a would-be contributor. Everything below was
reproduced on Linux, Python 3.11, CPU-only build.

Legend: 🔴 bug / broken promise · 🟠 missing feature that blocks a common workflow ·
🟡 paper cut / UX gap · ✅ worked well

---

## Stage 0 — install & first build

- ✅ `python -m venv` + `pip install -e .` works first try, ~37 s, no warnings.
- 🔴 **The default build type is `Debug`** (`[tool.scikit-build.cmake] build-type = "Debug"`
  in `pyproject.toml`). Measured on the same machine:

  | op | Debug | Release | slowdown |
  |---|---|---|---|
  | 512×512 matmul | 146.6 ms | 69.1 ms | 2.1× |
  | 1M-elem `x*y+x` | 13.86 ms | 0.29 ms | **48×** |

  Anyone who follows the README's build-from-source path and then benchmarks gets Debug
  numbers, which undercuts the "up to 3.5× faster than PyTorch" headline. Default should
  be `Release` (contributors can opt into Debug).
- 🟡 README install pointer (`.../expanded_assets/v0.1.0`) targets a release tag that
  doesn't exist; version badge says 0.0.0. Already tracked in #97/#98.

## Stage 1 — the first five minutes

- 🔴 `randn(mean, var, shape)`: the second parameter is named `var` but is used as the
  **standard deviation** (`randn(0, 4, [50000])` produces sample std ≈ 4.0, not 2.0).
  Either the name or the implementation is wrong; everything built on it (Linear init)
  inherits the confusion.
- 🟠 No `manual_seed` — no way to reproduce a run, and no way to write a
  parity test against PyTorch with fixed inputs short of hand-feeding lists.
- 🟠 No `arange`; no scalar construction (`Variable(5.0)` is a TypeError — you must
  write `Variable([5.0])`).
- 🟡 No `rsl.tensor(...)` alias; torch muscle memory hits `AttributeError` immediately.
- 🟡 `dtype`/`device` must be enum values (`rsl.float32`, `rsl.device.cpu`); string
  spellings like `dtype="float32"` are rejected with a long pybind signature dump.
- 🟡 No NumPy interop in either direction (no `__array__`/buffer protocol; constructor
  only takes nested Python lists). Fine as a zero-dependency stance, but worth a
  documented `from_numpy`/`numpy()` pair eventually.

## Stage 2 — inspecting values

- 🔴 `tolist()` returns a **flat** list regardless of shape — `[[1,2],[3,4]].tolist()`
  gives `[1.0, 2.0, 3.0, 4.0]`. Round-tripping a tensor through Python loses its shape.
- 🟠 No `item()`, `float(v)` fails, `len(v)` fails, iteration fails, `v[0]` fails. The
  only element accessor is the awkward `v.data.index([i, j])` → scalar.
- 🟡 `repr` prints a flat, truncated data list. It stays bounded for huge tensors
  (good), but there's no shape-aware formatting like numpy/torch, which makes eyeballing
  a 2-D result genuinely hard.

## Stage 3 — doing math

- 🟠 **No indexing or slicing of any kind** (`a[0]`, `a[:, -1]` → "not subscriptable").
  Tracked as a 2.0 blocker in #89.
- 🟠 **`matmul` and `transpose` are strictly 2-D** on both CPU and CUDA. Tracked in #89
  (batched matmul). Knock-on found while walking: `nets.layers.Linear` therefore
  rejects any batched-sequence input `(B, T, C)`.
- 🟠 Reductions: axis is mandatory (`a.sum()` TypeError — already in #89), negative
  axes rejected (`a.sum(-1)` TypeError, while `Softmax(dim=-1)` accepts them — the
  Python and C++ layers disagree), no `mean`, no keepdim control, and `max/min` return
  values only — **there is no argmax**, so you cannot compute classification accuracy
  without dumping to Python lists.
- 🟡 Error UX: every `LMP_CHECK` failure is printed to stderr *and* raised (each error
  appears twice), and messages leak absolute build-machine paths
  (`/home/user/rushlite/csrc/...`). Also `(2,2) @ (3,)` reports "Both matrices must be
  2D", which points at the wrong problem.
- 🟡 Comparisons return `float32`, not `bool`. Works fine as multiplicative masks;
  surprising coming from anywhere else.

## Stage 4 — training (README MLP example, run verbatim)

- 🔴 **The README training example barely learns**: loss went 980 → 949 → 935 → back up
  to 954 over 300 steps. Root cause is `Linear`'s init: `randn(0, 1)` — std 1.0 with no
  fan-in scaling, so a 256-wide layer saturates the softmax from step 0. Rescaling the
  same weights by 1/√fan_in makes the identical loop decrease monotonically. `Linear`
  needs Kaiming/Xavier-style default init.
- 🔴 **`Dropout` is inverted**: `(mask < p) * x` keeps activations with probability
  `p`. Measured: `Dropout(0.2)` keeps 20.3% of values — it drops 80% where torch drops
  20%. It also never rescales by 1/(1−p), and there is no `train()`/`eval()` mode on
  `Module`, so dropout stays on during inference.
- 🔴 **The `grad_fn` accessor is broken**: `y.grad_fn` raises
  `TypeError: Unable to convert function return value...` (pybind can't convert the
  raw `std::weak_ptr<Function>`). First thing a user prints when learning the autograd
  graph, and it crashes. (`repr(y)` renders it fine, so it's just the property binding.)
- 🟠 No optimizers (`optim.SGD`/`AdamW`), no loss functions, no `detach()`, no
  `state_dict`/save/load — a trained model cannot be checkpointed at all.
- 🟡 The functional update pattern (`model.params[name] = Variable(p.data - lr*p.grad,
  requires_grad=True)`) is workable and well documented in `Module.py`, but verbose,
  and forgetting the `requires_grad=True` rewrap silently freezes a parameter.
- 🟡 `Sequential` only accepts a list; torch-style varargs `Sequential(a, b)` raises.
- ✅ `no_grad` works correctly; grad accumulation + `zero_grad` behave; `backward()` on
  non-scalars works as advertised; broadcasting (incl. `(T,T)` vs `(B,T,T)`) is solid;
  `_Tensor` arithmetic for optimizer math (#92) works.

## Stage 5 — kernel fusion (the flagship)

- ✅ `capture_on`/`capture` run cleanly, including with unfused ops (matmul, softmax)
  inside the region — no crashes, graceful eager execution.
- 🟡 There is **no way to observe what was actually fused**. Fusion currently covers
  elementwise-binary only (#74), so most captured regions silently fall back to eager —
  and nothing tells the user. The flagship feature needs an introspection hook
  (`explain=True`, an env var, or a `last_fused_graph()` debug API) so users can verify
  it's doing anything, and so #75 benchmarks have something to assert against.

## Stage 6 — contributor experience

- ✅ `pytest tests/unit` passes out of the box (29 tests, 0.05 s)…
- 🟠 …but that only covers `grad_mode` and `Module`. The real correctness suites
  (`tests/graphs`, `tests/stress`) `import torch`, and torch is only an *optional*
  extra (`cpu`/`cu128`) behind uv-specific index config — a fresh contributor following
  the README's `uv sync` can't run them and gets an ImportError with no guidance.
  Related to #88 (contributor docs).

## Stage 7 — writing a real training script end-to-end

Wrote a standard 3-class spiral classifier (2→32→32→3 MLP, minibatch SGD,
cross-entropy, accuracy, checkpoint) three times: first as a torch user would write it,
then adapting around each wall. Final verdict: **it works — 99% accuracy in 2.6 s —
but every stage of the standard loop needs a workaround.**

1. **Batching**: the idiomatic `perm = randperm(len(X)); xb = X[perm[s:s+32]]` dies at
   `len()` and again at `X[idx]` (no fancy indexing, no slicing). Workaround: keep the
   dataset as Python lists and construct a fresh `Variable` per minibatch. Fine at
   spiral scale; a real dataset re-tokenizes Python floats every step.
2. **Cross-entropy**: the standard gather `probs[arange(n), y]` is impossible (no
   integer indexing), so targets must be one-hot on the host:
   `-(onehot * log(softmax(logits) + 1e-9)).sum(1).sum(0) / n`. Note `sum` keeps dims,
   so the "scalar" loss has shape `[1, 1]` and reading it is `loss.tolist()[0]`.
3. **Init, again**: out of the box this model sat at chance (~0.33). Rescaling weights
   by √(2/fan_in) and zeroing biases by hand is mandatory boilerplate today.
4. **Dropout in anger**: added `Dropout(0.1)` exactly as a torch user would →
   accuracy collapses 0.99 → **0.60** (it keeps 10% instead of dropping 10%). This is
   the inverted-mask bug from Stage 4 showing up in a realistic workflow; combined with
   no `eval()` mode, Dropout is currently unusable.
5. **Hand-rolled Adam**: doable, with three frictions — `p.grad` is a `_Tensor` so each
   use needs a `rsl.Variable(p.grad)` rewrap to reach the ops; the whole `step()` must
   sit inside `no_grad()` or optimizer math builds autograd graphs; and mixed
   `Variable`/`_Tensor` expressions force `.data` hops (e.g.
   `p.data - lr * (mhat / (sqrt(vhat) + eps)).data`).
6. **Accuracy**: no argmax, and `tolist()` is flat, so the metric is a Python loop over
   `flat[i*k:(i+1)*k].index(max(...))`. Slow and ugly, but the only route.
7. **Checkpointing**: no `state_dict`/save/load. Hand-rolled JSON works, but because
   `tolist()` loses shape you must store `p.shape` alongside the flat data and
   `reshape` on load. Round-trip verified (accuracy identical after reload).
8. **Unbatched inference**: `Linear` rejects a 1-D input `(features,)` because matmul
   is strictly 2-D — every single-sample call needs a manual `[[...]]` wrap and a
   `squeeze(0)` after.

---

## Roll-up: gaps not yet tracked in any open issue

Already tracked: batched matmul, slicing, Embedding, cat (#89) · conv/pool (#80) ·
fusion op coverage (#74) · benchmarks (#75) · README fixes (#97, #98) · wheels (#31).

New from this walk, roughly in priority order:

1. **Correctness bugs**: Dropout inverted + no eval mode; `randn` var/std misnaming;
   `Linear` unscaled init (kills the README example); `grad_fn` accessor pybind bug.
2. **Default build type → Release** (48× elementwise gap for source installs).
3. **Ergonomics with no C++ cost**: nested `tolist()`, `item()`/`__float__`,
   `sum()`/`mean()` all-reduce, negative axes at the binding layer, `Sequential`
   varargs, `manual_seed`, `arange`, string dtypes.
4. **Training loop table stakes**: `optim` module (SGD/AdamW — expressible in Python
   today), loss functions, `argmax`, `detach()`, `state_dict`/save/load,
   `Module.train()/eval()`.
5. **Fusion introspection** (see Stage 5) — prerequisite for credible #75 numbers.
6. **Error message polish**: raise-once (don't also print), relative paths, and
   shape-specific matmul diagnostics.
