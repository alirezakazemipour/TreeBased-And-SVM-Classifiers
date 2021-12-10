import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


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
    Y = data[:, -1].astype(int)
    full_batch_size = X.shape[0]
    n_class = np.max(Y)
    cv_num = 5
    max_tree = 50
    seed = 123
    history = {"train_acc": [], "val_acc": []}
    best_n_tree = None
    best_val_acc = 0

    for tree in tqdm(range(1, 1 + max_tree)):
        avg_val_acc = 0
        avg_train_acc = 0

        for x_train, y_train, x_val, y_val in choose_fold(X, Y, cv_num):
            clf = RandomForestClassifier(n_estimators=tree,
                                         criterion="entropy",
                                         random_state=seed,
                                         max_depth=7,  # best depth of DT from previous section
                                         bootstrap=False)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_train)
            avg_train_acc += (np.sum(y_pred == y_train) / len(y_pred)) * 100
            y_pred = clf.predict(x_val)
            avg_val_acc += (np.sum(y_pred == y_val) / len(y_pred)) * 100

        history["train_acc"].append(avg_train_acc / cv_num)
        history["val_acc"].append(avg_val_acc / cv_num)

        if history["val_acc"][-1] > best_val_acc:
            best_val_acc = history["val_acc"][-1]
            best_n_tree = tree

    print("best no. tree: {}, best val acc: {:.2f}".format(best_n_tree, best_val_acc))
    plt.plot(range(max_tree), history["train_acc"], c="r")
    plt.plot(range(max_tree), history["val_acc"], c="b")
    plt.legend(history.keys())
    plt.grid()
    plt.show()

    # clf = DecisionTreeClassifier(criterion="entropy", random_state=seed, max_depth=best_depth)
    # clf.fit(x_train, y_val)
    # y_pred = clf.predict(x_train)
    # con_mat = confusion_matrix(y_val, y_pred)
    # print(con_mat)
    # con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    # con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_class)], columns=[i for i in range(n_class)])
    # figure = plt.figure(figsize=(n_class, n_class))
    # sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('confusion matrix')
    # plt.show()
    # report = classification_report(y_val, y_pred)
    # print(report)
