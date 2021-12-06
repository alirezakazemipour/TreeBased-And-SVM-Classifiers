
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
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
    Y = data[:, -1].astype(int)
    # print(X.shape)
    # print(Y.shape)
    cv_num = 5
    seed = 123
    for i in range(cv_num):
        x_val = X[i * len(X) // cv_num: (i + 1) * len(X) // cv_num]
        y_val = Y[i * len(Y) // cv_num: (i + 1) * len(Y) // cv_num]

        x_train = np.delete(X, range(i * len(X) // cv_num, (i + 1) * len(X) // cv_num), axis=0)
        y_train = np.delete(Y, range(i * len(Y) // cv_num, (i + 1) * len(Y) // cv_num), axis=0)

        clf = DecisionTreeClassifier(criterion="entropy", random_state=seed)
        clf.fit(x_train, y_train)
        results = clf.predict(x_val)
        # print('predictions: ', results)
        # print("test labels: ", list(np.unique(y_val)))
        print(f'Accuracy: {(np.sum(results == y_val) / len(results)) * 100: .2f}%')
        # print(confusion_matrix(y_val, results))
