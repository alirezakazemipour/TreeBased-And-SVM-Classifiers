from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


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

    clfs = {"Linear SVM": {"base_clf": SVC(decision_function_shape="ovo",
                                           random_state=seed),
                           "param": "C",
                           "param_values": [1, 10]
                           },
            "RBF-SVM": {"base_clf": SVC(kernel="rbf",
                                        decision_function_shape="ovo",
                                        random_state=seed),
                        "param": "C",
                        "param_values": [1, 10, 100, 1000, 1e+4, 1e+5]
                        },
            "Decision Tree": {"base_clf": DecisionTreeClassifier(criterion="entropy", random_state=seed),
                              "param": "max_depth",
                              "param_values": np.arange(1, 21)
                              },
            "Random Forest": {"base_clf": RandomForestClassifier(criterion="entropy",
                                                                 random_state=seed,
                                                                 bootstrap=False),
                              "param": "n_estimators",
                              "param_values": np.arange(1, 46)
                              },
            "XGBoost": {"base_clf": XGBClassifier(booster="gbtree",
                                                  tree_method="hist",
                                                  reg_lambda=1e-3,
                                                  random_state=seed,
                                                  objective='logloss',
                                                  use_label_encoder=True,
                                                  verbosity=0),
                        "param": "learning_rate",
                        "param_values": [1, 5e-1, 1e-2, 5e-3]
                        }
            }
    for clf_name, config in clfs.items():
        history = {"train_acc": [], "val_acc": []}
        best_val_acc = 0
        best_param = None

        for p in tqdm(config["param_values"]):
            avg_val_acc = 0
            avg_train_acc = 0

            for x_train, y_train, x_val, y_val in choose_fold(X, Y, cv_num):
                clf = config["base_clf"]
                setattr(clf, config["param"], p)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_train)
                avg_train_acc += (np.sum(y_pred == y_train) / len(y_pred)) * 100
                y_pred = clf.predict(x_val)
                avg_val_acc += (np.sum(y_pred == y_val) / len(y_pred)) * 100

            history["train_acc"].append(avg_train_acc / cv_num)
            history["val_acc"].append(avg_val_acc / cv_num)

            if history["val_acc"][-1] > best_val_acc:
                best_val_acc = history["val_acc"][-1]
                best_param = p

        print(f"==> {clf_name} <==")
        print("Training result:")
        print("\tbest param: {} = {}\n\tbest validation accuracy = {:.2f}%".format(config["param"],
                                                                                   best_param,
                                                                                   best_val_acc))
        plt.plot(config["param_values"], history["train_acc"], c="r")
        plt.plot(config["param_values"], history["val_acc"], c="b")
        plt.legend(history.keys())
        plt.grid()
        plt.title(clf_name)
        plt.ylabel("Accuracy")
        plt.xlabel(config["param"])
        if "SVM" in clf_name or clf_name == "XGBoost":
            plt.xscale("log")
        plt.show()
        print("Test result: ")
        setattr(clf, config["param"], best_param)
        clf.fit(X, Y)
        y_pred = clf.predict(x_test)
        con_mat = confusion_matrix(y_test, y_pred)
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

        if clf_name == "Decision Tree":
            setattr(clfs["Random Forest"]["base_clf"], "max_depth", best_param)
            setattr(clfs["XGBoost"]["base_clf"], "max_depth", best_param)

        if clf_name == "Random Forest":
            setattr(clfs["XGBoost"]["base_clf"], "n_estimators", best_param)
