"""
This model cannot handel the problem properly, they are just build for fun so I free the random state variable and create different split datasets for more variance.
"""

import collections
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from USYD_data_preprocessing import USYD_data


seed = 0

acc = []
pred_y = collections.defaultdict(int)

lr_list = [i / 100.0 for i in range(5, 50)]

for lr in lr_list:
    # split training and validation
    x_train, x_test, y_train, y_test = USYD_data.get_data()
    clf = Perceptron(
        tol=1e-4,
        #  verbose=True,
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    acc.append(clf.score(x_test, y_test))
    cur_y = clf.predict(USYD_data.pred_x)
    pred_class = USYD_data.Res_remapping[cur_y[0]]
    pred_y[pred_class] += 1
    print(
        f"test for learning_rate {lr}, prediction is {pred_class}, with accuracy {acc[-1]}"
    )


# prediction
print(pred_y)

plt.plot(lr_list, acc, "-", color="#efc092")
plt.xlabel("learning rate of perceptron model")
plt.ylabel("accuracy for test set")

plt.show()
