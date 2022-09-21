import random

x = [random.randint(0, 10) for _ in range(10)]
W = [[random.randint(0, 10) for _ in range(6)] for _ in range(10)]

y = [0 for _ in range(len(W[0]))]  # x * W
for i in range(len(W[0])):
    for j in range(len(W)):
        y[i] += x[j] * W[j][i]
print(y)
