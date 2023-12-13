import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

delete_cols = ["LastPaymentOn","CurrentDebtDaysPrimary","DebtOccuredOn","CurrentDebtDaysSecondary",
               "DebtOccuredOnForSecondary","ExpectedLoss","LossGivenDefault","ExpectedReturn","ProbabilityOfDefault",
               "PrincipalOverdueBySchedule","PlannedPrincipalPostDefault","PlannedInterestPostDefault","EAD1","EAD2",
               "PrincipalRecovery","InterestRecovery","RecoveryStage","StageActiveSince","ModelVersion","Rating",
               "EL_V0","Rating_V0","EL_V1","Rating_V1","Rating_V2","Restructured","ActiveLateCategory",
               "WorseLateCategory","CreditScoreEsMicroL","CreditScoreEsEquifaxRisk",
               "CreditScoreFiAsiakasTietoRiskGrade","CreditScoreEeMini","PrincipalPaymentsMade",
               "InterestAndPenaltyPaymentsMade","PrincipalWriteOffs","InterestAndPenaltyWriteOffs","PrincipalBalance",
               "InterestAndPenaltyBalance","NoOfPreviousLoansBeforeLoan","AmountOfPreviousLoansBeforeLoan",
               "PreviousRepaymentsBeforeLoan","PreviousEarlyRepaymentsBefoleLoan",
               "PreviousEarlyRepaymentsCountBeforeLoan","GracePeriodStart","GracePeriodEnd","NextPaymentDate",
               "NextPaymentNr","NrOfScheduledPayments","ReScheduledOn","PrincipalDebtServicingCost",
               "InterestAndPenaltyDebtServicingCost","ActiveLateLastPaymentCategory","FirstPaymentDate",
               "MaturityDate_Original","MaturityDate_Last","PlannedInterestTillDate","RefinanceLiabilities"]

data = pd.read_csv('LoanData.csv')
for column in delete_cols:
    del data[column]
data = data[data["Status"] != "Current"]

data = data.dropna(axis=1, thresh=len(data)*0.5)
data = data.dropna()

cat_columns = data[['Country', 'EmploymentDurationCurrentEmployer', 'UseOfLoan',
                    'MaritalStatus', 'EmploymentStatus', 'OccupationArea', 'HomeOwnershipType']]
cat_columns_one_hot = pd.get_dummies(cat_columns).astype(int)


model_data = data[[
       'ApplicationSignedHour', 'ApplicationSignedWeekday', 'VerificationType', 'Amount', 'Interest',
       'LoanDuration', 'MonthlyPayment', 'Education',
        'IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
       'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal', 'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay']]
model_data = pd.merge(model_data.reset_index(), cat_columns_one_hot.reset_index())
del model_data["index"]

num_and_status = data.select_dtypes(include=['number'])
num_and_status["loan_status"] = data["Status"].map({"Repaid":0, "Late":1})


for feature_number in [10, 15, 20]:
    for selection_method in ["PCA", "KBEST"]:

        print(f"===={selection_method}={feature_number}====")

        with open(f"gridsearch_bondora_no_final_nn_{selection_method}_{feature_number}_lc.csv", "w") as f:
            f.write("fold;reduction;features;layers;activation;optimizer;batch;auc;precision;recall;acc\n")
        with open(f"gridsearch_bondora_no_final_xgboost_{selection_method}_{feature_number}_lc.csv", "w") as f:
            f.write("fold;reduction;features;eta;gamma;max_depth;lambda;alpha;auc;precision;recall;acc\n")
        with open(f"gridsearch_bondora_no_final_adaboost_{selection_method}_{feature_number}_lc.csv", "w") as f:
            f.write("fold;reduction;features;estimator;n_estimators;learning_rate;auc;precision;recall;acc\n")

        curr_model_data = model_data.to_numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=21)
        for fold, (train_index, test_index) in enumerate(kf.split(curr_model_data, num_and_status["loan_status"]), 1):
            print(f"Fold {fold}")
            X_train, X_test = curr_model_data[train_index, :], curr_model_data[test_index, :]
            y_train, y_test = num_and_status["loan_status"].iloc[train_index],  num_and_status["loan_status"].iloc[test_index]

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

            #condition = np.any(np.abs(X_train) > 3, axis=1)
            #X_train = X_train[~condition]
            #y_train = y_train[~condition]

            oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            X_over, y_over = oversampler.fit_resample(X_train, y_train)

            y_over = to_categorical(y_over)


            params = {
                "layers": [
                           (32,32,16),
                           (64,32,16),
                           (32,16),
                           (64,64,32)],
                "activation": ["tanh", "sigmoid"],
                "optimizer": ["adam", "sgd"],
                "batch": [64],
                "dropout": [None],
                "data": [(X_over, y_over)]
            }


            with open(f"gridsearch_bondora_no_final_nn_{selection_method}_{feature_number}_lc.csv", "a") as f:
                for param_set in list(product(*params.values())):

                    layers = param_set[0]
                    activation = param_set[1]
                    optimizer = param_set[2]
                    batch = param_set[3]
                    dropout = param_set[4]
                    X_data, y_data = param_set[5]

                    model = Sequential()
                    for i,layer in enumerate(layers):
                        if i == 0:
                            model.add(Dense(units=layer, input_dim=X_data.shape[1], activation=activation))
                        else:
                            model.add(Dense(units=layer, activation=activation))
                        if dropout:
                            model.add(Dropout(dropout))

                    model.add(Dense(units=2, activation=activation))

                    model.compile(loss="binary_crossentropy",
                          optimizer=optimizer,
                          metrics=['accuracy'])
                    model.fit(X_data, y_data, epochs=2, batch_size=batch, validation_data=(X_test, to_categorical(y_test)))

                    y_pred = model.predict(X_test)
                    conf_matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))

                    print("--------------------")
                    print(conf_matrix)

                    precision = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[0,1])
                    recall = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[1,0])
                    f1 = (2*conf_matrix[1,1])/(2*conf_matrix[1,1] + conf_matrix[0,1] + conf_matrix[1,0])

                    print("Precision", precision)
                    print("Recall", recall)
                    print("F1", f1)

                    auc = roc_auc_score(y_test, (y_pred[:,1]+1)/2)

                    print("auc", auc)

                    accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))

                    #model.save(f'nn_{layers}_{batch}.keras')
                    f.write(f"{fold};{selection_method};{feature_number};{layers};{activation};{optimizer};{batch};{auc};{precision};{recall};{accuracy}\n")

            params = {
                "eta": [0.1, 0.02],
                "gamma": [0.1, 1, 2],
                "max_depth": [4, 6],
                "lambda": [0, 1],
                "alpha": [0],
                "data": [(X_over, y_over)]
            }

            print(len(list(product(*params.values()))))
            with open(f"gridsearch_bondora_no_final_xgboost_{selection_method}_{feature_number}_lc.csv", "a") as f:
                for param_set in list(product(*params.values())):

                    X_data, y_data = param_set[5]

                    model = XGBClassifier(eta = param_set[0],
                                          gamma = param_set[1],
                                          max_depth = param_set[2],
                                          reg_lambda=param_set[3],
                                          alpha =param_set[4])

                    model.fit(X_data, y_data.argmax(axis=1))

                    print(model.score(X_test, y_test))

                    y_pred = model.predict(X_test)
                    conf_matrix = confusion_matrix(y_test, y_pred)

                    print(conf_matrix)

                    precision = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[0,1])
                    recall = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[1,0])
                    f1 = (2*conf_matrix[1,1])/(2*conf_matrix[1,1] + conf_matrix[0,1] + conf_matrix[1,0])

                    print("Precision", precision)
                    print("Recall", recall)
                    print("F1", f1)
                    print("---------------")

                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred)
                    #model.save_model(f'xgb_{param_set[3]}_{param_set[4]}.json')
                    f.write(f"{fold};{selection_method};{feature_number};{param_set[0]};{param_set[1]};{param_set[2]};{param_set[3]};{param_set[4]};{auc};{precision};{recall};{accuracy}\n")


            params = {
                "estimator": [DecisionTreeClassifier(max_depth=5)],
                "n_estimators": [50, 25],
                "learning_rate": [0.1, 0.2],
                "data": [(X_over, y_over)]
            }

            print(len(list(product(*params.values()))))
            with open(f"gridsearch_bondora_no_final_adaboost_{selection_method}_{feature_number}_lc.csv", "a") as f:
                for param_set in list(product(*params.values())):

                    X_data, y_data = param_set[3]

                    model = AdaBoostClassifier(estimator = param_set[0],
                                          n_estimators = param_set[1],
                                          learning_rate = param_set[2])

                    model.fit(X_data, y_data.argmax(axis=1))

                    print(model.score(X_test, y_test))

                    y_pred = model.predict(X_test)
                    conf_matrix = confusion_matrix(y_test, y_pred)

                    print(conf_matrix)

                    precision = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[0,1])
                    recall = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[1,0])
                    f1 = (2*conf_matrix[1,1])/(2*conf_matrix[1,1] + conf_matrix[0,1] + conf_matrix[1,0])

                    print("Precision", precision)
                    print("Recall", recall)
                    print("F1", f1)
                    print("---------------")

                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred)

                    #with open(f'ada_{param_set[2]}.pkl','wb') as pf:
                    #    pickle.dump(model,pf)

                    f.write(f"{fold};{selection_method};{feature_number};{param_set[0]};{param_set[1]};{param_set[2]};{auc};{precision};{recall};{accuracy}\n")
