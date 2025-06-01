import numpy as np
from tensor import Tensor, Dot

x = np.random.randn(2, 3)
w = np.random.randn(3, 2)
out = x.dot(w)
print(out)