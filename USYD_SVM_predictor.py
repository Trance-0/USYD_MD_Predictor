"""
This is the support vector machine model
The best performance is for linear kernel model
With accuracy about 0.85
You can change the c value, maybe you can have better result but I don't want to do so
"""

import numpy as np
from sklearn import svm
from USYD_data_preprocessing import USYD_data

from sklearn import metrics
import pandas as pd

seed = 0

data=USYD_data()

# split training and validation
x_train, x_test, y_train, y_test = data.get_data()

f_set = ["linear", "poly", "rbf", "sigmoid"]

for f in f_set:
    # clf stands for classifier
    clf = svm.SVC(kernel=f)
    clf.fit(x_train, y_train)

    # evaluate accuracy of model
    y_pred = clf.predict(x_test)
    print(
        f"Accuracy for kernel {f} using y_test:{metrics.accuracy_score(y_test, y_pred)}"
    )
    print(f"Decision function: {clf.decision_function([[1 for _ in range(7)]])}")
    print(f"Prediction: {data.Res_remapping[clf.predict(data.pred_x)[0]]}")
