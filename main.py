import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def choose_fold(x, y, n):
    for i in range(n):
        x_val = x[i * full_batch_size // cv_num: (i + 1) * full_batch_size // cv_num]
        y_val = y[i * full_batch_size // cv_num: (i + 1) * full_batch_size // cv_num]

        x_train = np.delete(x, range(i * full_batch_size // cv_num, (i + 1) * full_batch_size // cv_num), axis=0)
        y_train = np.delete(y, range(i * full_batch_size // cv_num, (i + 1) * full_batch_size // cv_num), axis=0)

        yield x_train, y_train, x_val, y_val


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
    Y = data[:, -1].astype(int) - 1
    n_class = np.max(Y) + 1
    cv_num = 5
    seed = 123

    Cs = [1, 10, 100, 1000, 1e+4, 1e+5]

    history = {"train_acc": [], "val_acc": []}
    best_c = None
    best_val_acc = 0

    np.random.seed(seed)
    shuffler = np.random.permutation(len(X))  # shuffling the dataset
    X = X[shuffler]
    Y = Y[shuffler]

    scaler = StandardScaler()  # essential for SVM
    X = scaler.fit_transform(X)

    test_idx = np.random.permutation(int(0.1 * len(X)))  # shuffling the dataset
    x_test = X[test_idx]
    y_test = Y[test_idx]

    X = np.delete(X, test_idx, axis=0)
    Y = np.delete(Y, test_idx, axis=0)
    full_batch_size = X.shape[0]

    for c in tqdm(Cs):
        avg_val_acc = 0
        avg_train_acc = 0

        for x_train, y_train, x_val, y_val in choose_fold(X, Y, cv_num):
            clf = SVC(C=c,
                      kernel="rbf",
                      decision_function_shape="ovo",
                      random_state=seed
                      )  # , gamma=1e-4, kernel="rbf") # linear -> acc = 87.28, rbf -> acc = 89
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_train)
            avg_train_acc += (np.sum(y_pred == y_train) / len(y_pred)) * 100
            y_pred = clf.predict(x_val)
            avg_val_acc += (np.sum(y_pred == y_val) / len(y_pred)) * 100

        history["train_acc"].append(avg_train_acc / cv_num)
        history["val_acc"].append(avg_val_acc / cv_num)

        if history["val_acc"][-1] > best_val_acc:
            best_val_acc = history["val_acc"][-1]
            best_c = c

    print("best c: {}, best val acc: {:.2f}".format(best_c, best_val_acc))
    plt.plot(range(len(Cs)), history["train_acc"], c="r")
    plt.plot(range(len(Cs)), history["val_acc"], c="b")
    plt.legend(history.keys())
    plt.grid()
    plt.show()

    clf = SVC(kernel="rbf",
              decision_function_shape="ovo",
              random_state=seed,
              C=best_c
              )
    clf.fit(X, Y)
    y_pred = clf.predict(x_test)
    con_mat = confusion_matrix(y_test, y_pred)
    # print(con_mat)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_class)], columns=[i for i in range(n_class)])
    figure = plt.figure(figsize=(n_class, n_class))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()
    report = classification_report(y_test, y_pred)
    print(report)
