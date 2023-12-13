import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('accepted_2007_to_2018Q4.csv')
data = data[data['application_type'] == "Individual"]
data = data[(data['loan_status'] == "Charged Off") | (data['loan_status'] == "Fully Paid") |
            (data['loan_status'] == "Default") | (data['loan_status'] == "Late (31-120 days)") |
            (data['loan_status'] == "Late (16-30 days)")]


data = data.dropna(axis=1, thresh=len(data)*0.5)
data = data.dropna()

counts = Counter(data['title'])
data['title'] = data['title'].map(lambda x: x if counts[x] > 9000 else "other")

cat_columns = data.select_dtypes(exclude=['number'])
cat_columns = cat_columns[['home_ownership', 'title']]

del data['policy_code']

numerical_columns = data.select_dtypes(include=['number'])
num_and_status = data[[*numerical_columns.columns, 'loan_status']]

mapping = {'Fully Paid': 0, 'Charged Off': 1, "Default": 1, "Late (31-120 days)": 1, "Late (16-30 days)": 1}
num_and_status['loan_status'] = num_and_status['loan_status'].map(mapping)

col_set = [
    "loan_amnt",
    "term",
    "int_rate",
    "emp_length",
    "annual_inc",
    "dti",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "avg_cur_bal",
    "earliest_cr_line"
]

model_data = data[col_set]

term_map = {" 36 months" : 1, " 60 months" : 0}
emp_map = {'10+ years': 10,
         '2 years': 2,
         '3 years': 3,
         '< 1 year': 0,
         '1 year': 1,
         '4 years': 4,
         '5 years': 5,
         '6 years': 6,
         '8 years': 8,
         '7 years': 7,
         '9 years': 9
          }
def earliest_cr_line_map(entry):
    return 1970 if int(entry[4:8]) < 1970 else 2010 if int(entry[4:8]) > 2010 else int(entry[4:8])

model_data['earliest_cr_line'] = model_data['earliest_cr_line'].apply(earliest_cr_line_map)
model_data['emp_length'] = model_data['emp_length'].map(emp_map)
model_data['term'] = model_data['term'].map(term_map)
model_data = model_data.reset_index()
del model_data["index"]



for feature_number in [3,10]:
    for selection_method in ["PCA", "KBEST"]:

        print(f"===={selection_method}={feature_number}====")

        with open(f"gridsearch_outlier_lc_cat_{selection_method}_{feature_number}_lc.csv", "w") as f:
            f.write("fold;reduction;features;iterations;depth;learning_rate;auc;precision;recall;acc\n")


        curr_model_data = model_data.to_numpy()
        curr_model_data = np.hstack((cat_columns.to_numpy(), curr_model_data))

        kf = KFold(n_splits=5, shuffle=True, random_state=21)
        for fold, (train_index, test_index) in enumerate(kf.split(curr_model_data, num_and_status["loan_status"]), 1):
            print(f"Fold {fold}")
            X_train, X_test = curr_model_data[train_index, :], curr_model_data[test_index, :]
            y_train, y_test = num_and_status["loan_status"].iloc[train_index],  num_and_status["loan_status"].iloc[test_index]

            X_train_cat_columns = X_train[:, :2]
            X_train = X_train[:, 2:]

            X_test_cat_columns = X_test[:, :2]
            X_test = X_test[:, 2:]

            x_scaler = StandardScaler()

            X_train = x_scaler.fit_transform(X_train)
            X_test = x_scaler.transform(X_test)

            if selection_method == "PCA":
                pca = PCA(n_components=feature_number)
                X_train = pca.fit_transform(X_train, y_train)
                X_test = pca.transform(X_test)
            else:
                kbest = SelectKBest(k=feature_number)
                X_train = kbest.fit_transform(X_train, y_train)
                X_test = kbest.transform(X_test)

            condition = np.any(np.abs(X_train) > 3, axis=1)
            X_train = X_train[~condition]
            X_train_cat_columns = X_train_cat_columns[~condition]
            y_train = y_train[~condition]

            X_train = np.hstack((X_train_cat_columns, X_train))
            X_test = np.hstack((X_test_cat_columns, X_test))

            print(X_train.shape)
            print(X_test.shape)

            oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            X_over, y_over = oversampler.fit_resample(X_train, y_train)

            y_over = to_categorical(y_over)

            params = {
                "iterations": [100, 50],
                "depth": [5, 10],
                "learning_rate": [0.1, 0.2]
            }

            with open(f"gridsearch_outlier_lc_cat_{selection_method}_{feature_number}_lc.csv", "a") as f:
                for param_set in list(product(*params.values())):
                    print(param_set)

                    model = CatBoostClassifier(cat_features=[0, 1], iterations=param_set[0],
                                               depth=param_set[1], learning_rate=param_set[2])
                    model.fit(X_over, y_over.argmax(axis=1), logging_level="Silent")

                    print(model.score(X_test, y_test))

                    y_pred = model.predict(X_test)
                    conf_matrix = confusion_matrix(y_test, y_pred)

                    print(conf_matrix)

                    precision = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1])
                    recall = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[1, 0])
                    f1 = (2 * conf_matrix[1, 1]) / (2 * conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])

                    print("Precision", precision)
                    print("Recall", recall)
                    print("F1", f1)
                    print("---------------")

                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)

                    print("auc", auc)

                    accuracy = accuracy_score(y_test, y_pred)
                    f.write(f"{fold};{selection_method};{feature_number};{param_set[0]};{param_set[1]};{param_set[2]};{auc};{precision};{recall};{accuracy}\n")

