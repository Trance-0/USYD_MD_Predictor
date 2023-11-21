"""
This model can have fantastic performance when node count is about 400.
You can modify the seed to any number.
Best accuracy is around 0.9
"""
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from USYD_data_preprocessing import USYD_data
import collections

# seed = 39

data=USYD_data()

# split training and validation
x_train, x_test, y_train, y_test = data.get_data()

acc = []
pred_y = collections.defaultdict(int)

node_counts = [i for i in range(8, 512, 8)]

best_acc = 0
best_nodecount = 0

for node in node_counts:
    mlp = MLPClassifier(
        activation="logistic",
        solver="adam",
        hidden_layer_sizes=(node,),
        alpha=1e-4,
        tol=1e-4,
        max_iter=8192,
        #    verbose=True,
        # random_state=seed,
        learning_rate_init=0.0005,
    )
    mlp.fit(x_train, y_train)

    # validation
    acc.append(mlp.score(x_test, y_test))
    cur_y = mlp.predict(data.pred_x)
    pred_class=data.Res_remapping[cur_y[0]]
    pred_y[pred_class] += 1
    print(f"test for node count {node}, prediction is {pred_class}, with accuracy {acc[-1]}")
    if acc[-1] > best_acc:
        best_nodecount = node
        best_acc = acc[-1]

plt.plot(node_counts, acc, "-", color="#efc092")
plt.xlabel("hidden layer node count")
plt.ylabel("accuracy for test set")
plt.xscale("log")

print(
    f"The best node count for the hidden layer for this sample is {best_nodecount}, with test accurary {best_acc}"
)
plt.show()


# prediction
print(pred_y)