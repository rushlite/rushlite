"""Attempt 3: + Dropout, + hand-rolled Adam, + checkpoint round-trip."""
import json
import math
import random

import rushlite as rsl
from rushlite.nets import layers

random.seed(0)

N, K = 100, 3
X, Y = [], []
for k in range(K):
    for i in range(N):
        r = i / N
        t = k * 4 + 4 * r + random.gauss(0, 0.2)
        X.append([r * math.sin(t), r * math.cos(t)])
        Y.append(k)

def build(with_dropout):
    net = layers.Sequential([
        layers.Linear(2, 32), layers.ReLU(),
    ] + ([layers.Dropout(0.1)] if with_dropout else []) + [
        layers.Linear(32, 32), layers.ReLU(),
        layers.Linear(32, K),
    ])
    for name, p in net.named_parameters():
        scale = (2.0 / p.shape[0]) ** 0.5 if len(p.shape) == 2 else 0.0
        net.params[name] = rsl.Variable(p.data * scale, requires_grad=True)
    return net

softmax = layers.Softmax(dim=-1)

def cross_entropy(logits, onehot, n):
    return -(onehot * rsl.log(softmax(logits) + 1e-9)).sum(1).sum(0) / n

def accuracy(model, X, Y):
    logits = model(rsl.Variable(X))
    flat, (n, k) = logits.tolist(), logits.shape
    return sum(
        flat[i * k:(i + 1) * k].index(max(flat[i * k:(i + 1) * k])) == Y[i]
        for i in range(n)) / n

# ---- Adam, hand-rolled on top of the params-rebind pattern ----
class Adam:
    def __init__(self, model, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps = model, lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self):
        self.t += 1
        with rsl.no_grad():
            for name, p in self.model.named_parameters():
                g = rsl.Variable(p.grad)          # _Tensor -> Variable to use ops
                if name not in self.m:
                    self.m[name] = g * 0.0
                    self.v[name] = g * 0.0
                self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
                self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * g * g
                mhat = self.m[name] / (1 - self.b1 ** self.t)
                vhat = self.v[name] / (1 - self.b2 ** self.t)
                upd = p.data - self.lr * (mhat / (rsl.sqrt(vhat) + self.eps)).data
                self.model.params[name] = rsl.Variable(upd, requires_grad=True)

def train(model, epochs=100, opt=None):
    for epoch in range(epochs):
        order = list(range(len(X)))
        random.shuffle(order)
        for s in range(0, len(order), 32):
            idx = order[s:s + 32]
            xb = rsl.Variable([X[i] for i in idx])
            oh = rsl.Variable([[1.0 if Y[i] == c else 0.0 for c in range(K)] for i in idx])
            loss = cross_entropy(model(xb), oh, len(idx))
            model.zero_grad()
            loss.backward()
            if opt:
                opt.step()
            else:
                for name, p in model.named_parameters():
                    model.params[name] = rsl.Variable(p.data - 0.5 * p.grad, requires_grad=True)

print("== SGD, no dropout ==")
m1 = build(False); train(m1)
print("acc:", round(accuracy(m1, X, Y), 3))

print("== SGD, Dropout(0.1) as a torch user would add it ==")
m2 = build(True); train(m2)
print("acc:", round(accuracy(m2, X, Y), 3), " <- dropout bug: keeps 10%, drops 90%")

print("== Adam, no dropout ==")
m3 = build(False); train(m3, opt=Adam(m3))
print("acc:", round(accuracy(m3, X, Y), 3))

# ---- checkpoint round-trip: no state_dict/save, so hand-roll via tolist ----
print("== checkpoint round-trip ==")
ckpt = {name: {"data": p.tolist(), "shape": p.shape}   # tolist is flat -> must store shape too
        for name, p in m3.named_parameters()}
json.dump(ckpt, open("ckpt.json", "w"))

m4 = build(False)
loaded = json.load(open("ckpt.json"))
for name, entry in loaded.items():
    flat, shape = entry["data"], entry["shape"]
    v = rsl.Variable(flat, requires_grad=True)        # 1-D
    m4.params[name] = rsl.Variable(rsl.reshape(v, shape).data, requires_grad=True)
print("acc after reload:", round(accuracy(m4, X, Y), 3))
