import matplotlib.pyplot as plt
import numpy as np

import nn

EPOCHS = 3000

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([0, 1, 1, 0])

model = nn.Linear(2, 1)
learner = nn.Learner(model, nn.mse_loss, nn.SGDOptimizer(lr=.05))
loss1 = learner.fit(X, Y, epochs=EPOCHS, bs=4)

model2 = nn.Sequential(
    nn.Linear(2, 15),
    nn.Sigmoid(),
    nn.Linear(15, 1)
)
learner2 = nn.Learner(model2, nn.mse_loss, nn.SGDOptimizer(lr=.02))
loss2 = learner2.fit(X, Y, epochs=EPOCHS, bs=4)

plt.plot(loss1)
plt.plot(loss2)
plt.show()


def accuracy(y_pred, y):
    return np.sum(y_pred == y) / len(y)


y_pred1, _ = model.forward(X)
y_pred1 = [1 if p > .5 else 0 for p in y_pred1.flatten()]

y_pred2, _ = model2.forward(X)
y_pred2 = [1 if p > .5 else 0 for p in y_pred2.flatten()]

print(accuracy(y_pred1, Y))
print(accuracy(y_pred2, Y))
