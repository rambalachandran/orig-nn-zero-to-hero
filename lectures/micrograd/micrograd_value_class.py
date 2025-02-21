# Test code for Value Class. Building it up slowly
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from  lectures.micrograd.plot_graph import draw_dot

# %%
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        # _children is a tuple of values with default being an empty tuple
        self.data = data
        self.grad = 0.0
        self._backward = (
            lambda: None
        )  # By default, the backward function does nothing, for example on the leaf nodes 
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # DONT GET: Why do we need to pass (self, other) to the __add__ method?
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad * 1.0  # dL/dx = dL/dy * dy/dx and dy/dx = 1
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __mul__(self, other):
        # The passing of (self, other) as child is key to be able to invoke the _backward() function and work as expected
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2) # d/dx tanh(x) = 1 - tanh(x)**2
        out._backward = _backward
        return out

 # %%   
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(1.0, label='x2')
x1x2 = x1 + x2

# %%
x1x2.grad = 1.0

# %%
x1x2._backward()

# %%
x2.data = 4.0
# %%
draw_dot(x1x2)

# %%
