# %%
import math
from typing import Any

from lectures.micrograd.plot_graph import draw_dot


class Value:
    def __init__(self, data, _children = (), _op = '', label='') -> None:
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None
        self.label = label
        self._op = _op
        self.topo = []

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        x = self.data + other.data
        out = Value(x, _children=(self, other),  _op='+')
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward
        return out
    
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        x = self.data * other.data
        out = Value(x, _children = (self, other), _op='*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        x = self.data ** other
        out = Value(x, _children=(self,), _op = '**')
        def _backward():
            self.grad += out.grad * other*(self.data**(other-1))
        out._backward = _backward
        return out
    
    def test_topo(self):
        # This function is to test topo to ensure the topo building works as expected
        recurse_children = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:                            
                build_topo(child)
            recurse_children.append(v)
        build_topo(self)
        # recurse_children.append(self)
        return recurse_children
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:                            
                build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.topo = reversed(topo)
        # Why is this grad being set to 1? Since the derivative by itself is 1
        self.grad = 1 
        # topo = reversed(topo)
        for node in self.topo:
            node._backward()
         
    # Copy pasted from previous try
    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

# %%
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
e = (2*n).exp()
o = (e - 1) / (e + 1)
o.label = 'o'
o.backward()



# %%
draw_dot(o)
# %%
class Neuron:
    def __init__(num_inputs: int) -> None:
        self.w = [Value(1) for i in range(num_inputs)]
    
    def __call__(self, x:list) -> Any:
        act = sum(xi*wi for (xi,wi) in zip(x, self.w))
        out = act.tanh()