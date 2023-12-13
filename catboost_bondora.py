from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
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

cat_columns = data[['Country','EmploymentDurationCurrentEmployer', 'UseOfLoan',
       'MaritalStatus', 'EmploymentStatus', 'OccupationArea', 'HomeOwnershipType']]

for int_col in ['UseOfLoan', 'MaritalStatus', 'EmploymentStatus', 'OccupationArea', 'HomeOwnershipType']:
    cat_columns[int_col] = cat_columns[int_col].astype(int)

num_and_status = data.select_dtypes(include=['number'])

model_data = data[[
       'ApplicationSignedHour', 'ApplicationSignedWeekday', 'VerificationType', 'Amount', 'Interest',
       'LoanDuration', 'MonthlyPayment', 'Education', 'IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
       'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal', 'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay']]
model_data = model_data.reset_index()
del model_data["index"]

num_and_status["loan_status"] = data["Status"].map({"Repaid":0, "Late":1})



for feature_number in [3,10,18]:
    for selection_method in ["PCA", "KBEST"]:

        print(f"===={selection_method}={feature_number}====")

        with open(f"gridsearch_bo_cat_outl_{selection_method}_{feature_number}_lc.csv", "w") as f:
            f.write("fold;reduction;features;iterations;depth;learning_rate;auc;precision;recall;acc\n")

        curr_model_data = model_data.to_numpy()
        curr_model_data = np.hstack((cat_columns.to_numpy(), curr_model_data))

        kf = KFold(n_splits=5, shuffle=True, random_state=21)
        for fold, (train_index, test_index) in enumerate(kf.split(curr_model_data, num_and_status["loan_status"]), 1):
            print(f"Fold {fold}")
            X_train, X_test = curr_model_data[train_index, :], curr_model_data[test_index, :]
            y_train, y_test = num_and_status["loan_status"].iloc[train_index],  num_and_status["loan_status"].iloc[test_index]

            X_train_cat_columns = X_train[:, :7]
            X_train = X_train[:, 7:]

            X_test_cat_columns = X_test[:, :7]
            X_test = X_test[:, 7:]

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

            with open(f"gridsearch_bo_cat_outl_{selection_method}_{feature_number}_lc.csv", "a") as f:
                for param_set in list(product(*params.values())):
                    print(param_set)
                    model = CatBoostClassifier(cat_features=list(range(7)), iterations=param_set[0],
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

