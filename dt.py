import numpy as np

data = []
with open("data.txt", 'r') as f:
    for i, line in enumerate(f):
        str_values = line.split(',')
        float_values = []
        for v in str_values:
            float_values.append(float(v))
        data.append(float_values)

data = np.stack(data)
X = data[:, 1:-1]
Y = data[:, -1].astype(np.uint)
print(X.shape)
print(Y.shape)