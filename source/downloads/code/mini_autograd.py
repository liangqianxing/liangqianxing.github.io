import numpy as np


def ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self.op!r})"

    def __matmul__(self, other):
        other = ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="matmul",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="relu",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=axis, keepdims=True)
        out = Tensor(
            probs,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="softmax",
        )

        def _backward():
            if self.requires_grad:
                dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self._accumulate_grad(probs * (out.grad - dot))

        out._backward = _backward
        return out

    def cross_entropy(self, target):
        target = np.asarray(target, dtype=np.int64)
        probs = np.clip(self.data, 1e-12, 1.0)
        batch_indices = np.arange(target.shape[0])
        loss_value = -np.log(probs[batch_indices, target]).mean()
        out = Tensor(
            loss_value,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="cross_entropy",
        )

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[batch_indices, target] = -1.0 / probs[batch_indices, target]
                grad /= target.shape[0]
                self._accumulate_grad(out.grad * grad)

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if grad is None:
            if self.data.shape != ():
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            for parent in tensor.parents:
                build_topo(parent)
            topo.append(tensor)

        build_topo(self)
        self.grad = grad

        for tensor in reversed(topo):
            tensor._backward()

    def zero_grad(self):
        self.grad = None

    def _accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad


def demo():
    np.random.seed(0)

    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
            ]
        )
    )
    y = np.array([0, 1, 1])

    w1 = Tensor(np.random.randn(3, 4) * 0.1, requires_grad=True)
    w2 = Tensor(np.random.randn(4, 2) * 0.1, requires_grad=True)

    for step in range(50):
        for parameter in (w1, w2):
            parameter.zero_grad()

        logits = (x @ w1).relu() @ w2
        probs = logits.softmax(axis=1)
        loss = probs.cross_entropy(y)
        loss.backward()

        learning_rate = 0.5
        w1.data -= learning_rate * w1.grad
        w2.data -= learning_rate * w2.grad

        if step % 10 == 0:
            print(f"step={step:02d}, loss={loss.data:.6f}")


if __name__ == "__main__":
    demo()
