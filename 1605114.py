import pandas as pd
from sklearn import preprocessing
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def telco_processing():
    global df_telco
    df_telco = pd.read_csv("D:/L-4 T-2/CSE472 Machine Learning Sessional/offline 1/adaboost/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df_telco.drop('customerID', inplace=True, axis=1)
    # encode column
    encode_array =  [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod']
    ord_enc = preprocessing.OrdinalEncoder()
    for x in encode_array:
        df_telco[x] = ord_enc.fit_transform(df_telco[[x]])

    df_telco['Churn'] = df_telco['Churn'].replace('Yes',1)
    df_telco['Churn'] = df_telco['Churn'].replace('No',-1)
    # replace  empty with nan and nan with mean
    df_telco['TotalCharges'] = df_telco['TotalCharges'].replace(r'^\s+$', np.nan, regex=True)
    df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'])
    df_telco['TotalCharges'].fillna((df_telco['TotalCharges'].mean()), inplace=True)
    # df_telco.fillna(df_telco.mean())

    scaler = preprocessing.StandardScaler().fit(df_telco[['MonthlyCharges']])
    df_telco['MonthlyCharges'] = scaler.transform(df_telco[['MonthlyCharges']])
    scaler = preprocessing.StandardScaler().fit(df_telco[['TotalCharges']])
    df_telco['TotalCharges'] = scaler.transform(df_telco[['TotalCharges']])

     # normalize(df):
    result = df_telco.copy()
    # for feature_name in df_telco.columns:
    #     max_value = df_telco[feature_name].max()
    #     min_value = df_telco[feature_name].min()
    #     result[feature_name] = (df_telco[feature_name] - min_value) / (max_value - min_value)

    df_telco = result
    # print(dataset.isna().sum()) # dataset having nan value
    # print(df_telco[450:490].head(20))
    # print(df_telco.columns)

def adult_data_processing():
    global df_adult_data
    global df_adult_data_test
    df_adult_data = pd.read_csv("D:/L-4 T-2/CSE472 Machine Learning Sessional/offline 1/adaboost/dataset/adult.data",
                                names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                       'hours-per-week', 'native-country', 'decision'])

    # numerical_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num']
    encode_array = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'native-country']

    df_adult_data['workclass'] = df_adult_data['workclass'].replace(' ?', np.nan)
    df_adult_data['occupation'] = df_adult_data['occupation'].replace(' ?', np.nan)
    df_adult_data['native-country'] = df_adult_data['native-country'].replace(' ?', np.nan)

    # df_telco['TotalCharges'].fillna((df_telco['TotalCharges'].mean()), inplace=True)

    df_adult_data['workclass'] = df_adult_data['workclass'].fillna(df_adult_data['workclass'].value_counts().idxmax())
    df_adult_data['occupation'] = df_adult_data['occupation'].fillna(
        df_adult_data['occupation'].value_counts().idxmax())
    df_adult_data['native-country'] = df_adult_data['native-country'].fillna(
        df_adult_data['native-country'].value_counts().idxmax())
    ord_enc = preprocessing.OrdinalEncoder()
    for x in encode_array:
        df_adult_data[x] = ord_enc.fit_transform(df_adult_data[[x]])

    continuous = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num']
    for x in continuous:
        scaler = preprocessing.StandardScaler().fit(df_adult_data[[x]])
        df_adult_data[x] = scaler.transform(df_adult_data[[x]])

    df_adult_data = df_adult_data.dropna()
    df_adult_data['decision'] = df_adult_data['decision'].replace(' >50K', 1)
    df_adult_data['decision'] = df_adult_data['decision'].replace(' <=50K', -1)

    df_adult_data_test = pd.read_csv(
        "D:/L-4 T-2/CSE472 Machine Learning Sessional/offline 1/adaboost/dataset/adult.test",
        names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'decision'])

    df_adult_data_test['workclass'] = df_adult_data_test['workclass'].replace(' ?', np.nan)
    df_adult_data_test['occupation'] = df_adult_data_test['occupation'].replace(' ?', np.nan)
    df_adult_data_test['native-country'] = df_adult_data_test['native-country'].replace(' ?', np.nan)

    # df_telco['TotalCharges'].fillna((df_telco['TotalCharges'].mean()), inplace=True)

    df_adult_data_test['workclass'] = df_adult_data_test['workclass'].fillna(
        df_adult_data_test['workclass'].value_counts().idxmax())
    df_adult_data_test['occupation'] = df_adult_data_test['occupation'].fillna(
        df_adult_data_test['occupation'].value_counts().idxmax())
    df_adult_data_test['native-country'] = df_adult_data_test['native-country'].fillna(
        df_adult_data_test['native-country'].value_counts().idxmax())
    ord_enc = preprocessing.OrdinalEncoder()
    for x in encode_array:
        df_adult_data_test[x] = ord_enc.fit_transform(df_adult_data_test[[x]])

    for x in continuous:
        # print(x)
        scaler = preprocessing.StandardScaler().fit(df_adult_data_test[[x]])
        df_adult_data_test[x] = scaler.transform(df_adult_data_test[[x]])

    df_adult_data_test = df_adult_data_test.dropna()
    df_adult_data_test['decision'] = df_adult_data_test['decision'].replace(' >50K.', 1)
    df_adult_data_test['decision'] = df_adult_data_test['decision'].replace(' <=50K.', -1)


# print("test..",df_test.shape)


def creditCard_processing():
    global df_credit
    df_credit = pd.read_csv(
        "D:/L-4 T-2/CSE472 Machine Learning Sessional/offline 1/adaboost/dataset/creditcard.csv")

    # print(df_credit.columns)
    df_credit.drop('Time', inplace=True, axis=1)
    features = [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    target = 'Class'

    positive = df_credit[df_credit["Class"] == 1]
    negative_all = df_credit[df_credit["Class"] == 0]
    negative = negative_all.sample(n=2000)

    frames = [positive, negative]
    df_credit = pd.concat(frames)
    # shuffle the DataFrame rows
    df_credit = df_credit.sample(frac=1)
    for x in features:
        # print(x)
        scaler = preprocessing.StandardScaler().fit(df_credit[[x]])
        df_credit[x] = scaler.transform(df_credit[[x]])

    df_credit['Class'] = df_credit['Class'].replace(0, -1)


def logisticRegression(X_train, Y_train ):
    learning_rate = 0.01
    iterations = 1000
    # m = row , n = column
    m, n = X_train.shape
    W = np.zeros(n)
    b = 0

    for i in range(iterations):
       # A =  (np.exp( X_train.dot(W) + b) - np.exp(- (X_train.dot(W) + b))) / (np.exp( X_train.dot(W) + b) + np.exp(- (X_train.dot(W) + b)))
        A = np.tanh( X_train.dot(W) + b)

        # dw = np.dot(X_train.T,(Y_train-A)*(1-A**2))
        dw=np.matmul(X_train.T,(A-Y_train))/m
        db=np.sum(A-Y_train)/m

        # update weights
        W = W - learning_rate * dw
        b = b - learning_rate * db

    return b , W

def lr_predict(X_test, b , W):
    Z = np.tanh(X_test.dot(W) + b)
    Y = np.where(Z > 0, 1, -1)
    return Y


def normalize(w):
    sum = 0
    for i in range(len(w)):
        sum += w[i]
    for i in range(len(w)):
        w[i] = w[i]/sum
    return w

def adaboost(data, l_weak,target, k):
    w = []
    h = []
    z = []
    for i in range(data.shape[0]):
        w.append(1.0 / data.shape[0])

    for i in range(k):
        resampled_data = data.sample(frac=1, weights=w, replace=True, random_state=1)

        X_train = resampled_data.drop([target], axis=1).values  # independant features
        Y_train = resampled_data[target].values  # dependant variable
        Y_test = data[target].values
        X_test = data.drop([target], axis=1).values

        b , W = l_weak(X_train, Y_train)
        h.append([b , W])
        pred = lr_predict(X_test,b,W)
        # print("predict ", pred)
        error = 0

        for j in range(len(pred)):
            if pred[j] != Y_test[j]:
                error = error + w[j]

        z.append(np.log2((1 - error) / error))
        if error > 0.5:
            continue
        for j in range(len(pred)):
            if pred[j] == Y_test[j]:
                w[j] = w[j]*(error/(1-error))

        w = normalize(w)

    return h,z


def adaboost_predict(data, h , z):
    predict = []
    # print(len(h))
    # weighted_array = [0.0 for i in range(len([data]))]
    # for i in range(len(h)):
    #     # print(h[i][0])
    #     # print(h[i][1])
    #     # print("CCCCCCCCCC")
    #     pred = lr_predict(data, h[i][0], h[i][1])
    #     an_array = np.array(pred)
    #     # print("Z : ", z)
    #     multiplied_array = an_array * z[i]
    #     weighted_array = np.add(weighted_array, multiplied_array)

    weighted_array = []
    for i in range(len(h)):
        pred = lr_predict(data, h[i][0], h[i][1])
        predict.append(pred)
    # print(len(predict[0]))
    for i in range(len(predict[0])):
        ws = 0.0
        for j in range(len(h)):
            ws = ws + predict[j][i] * z[j]
        ws = ((ws*1.0))
        if ws > 0.0 :
            weighted_array.append(1)
        else:
            weighted_array.append(-1)

    # weighted_array = weighted_array / len(z)
    # Y = np.where(weighted_array > 0, 1, -1)
    # print(weighted_array)
    return weighted_array


#
# ---------------------------------------- main-----------------------------------
#


print("...............................telco ..................................")
telco_processing()
train_df, test_df = train_test_split(df_telco, test_size=0.2, random_state=42, shuffle=True)
target = 'Churn'

print("Logistic Reg : ")
X_train = train_df.drop([target], axis=1).values  # independant features
Y_train = train_df[target].values  # dependant variable
Y_test = test_df[target].values
X_test = test_df.drop([target], axis=1).values
b , W = logisticRegression(X_train,Y_train)

pred = lr_predict(X_train,b,W)
tn, fp, fn, tp = confusion_matrix(Y_train,pred).ravel()
# print(tn,fp,fn,tp)
print("\n<<<  For train data : >>>\n")
print("acc : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

pred = lr_predict(X_test,b,W)
tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()
# print(tn,fp,fn,tp)
print("\n****For test data : ****\n")
print("Accuracy : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

print("\n___________Adaboost__________\n")
k = [5,10,15,20]
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_train,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_train,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()

print("\nFor test data : \n")
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_test,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_test,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()


print("\n\n...............................adult ..................................\n")
adult_data_processing()
train_df = df_adult_data
test_df = df_adult_data_test

target = 'decision'

print("Logistic Reg : ")
X_train = train_df.drop([target], axis=1).values  # independant features
Y_train = train_df[target].values  # dependant variable
Y_test = test_df[target].values
X_test = test_df.drop([target], axis=1).values
b , W = logisticRegression(X_train,Y_train)

pred = lr_predict(X_train,b,W)
tn, fp, fn, tp = confusion_matrix(Y_train,pred).ravel()
# print(tn,fp,fn,tp)
print("\n<<<  For train data : >>>\n")
print("Accuracy : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

pred = lr_predict(X_test,b,W)
tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()
# print(tn,fp,fn,tp)
print("\n****For test data : ****\n")
print("Accuracy : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

print("\n___________Adaboost__________\n")
k = [5,10,15,20]
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_train,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_train,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()

print("\nFor test data : \n")
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_test,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_test,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()




print("............................... credit  ..................................")
creditCard_processing()
train_df, test_df = train_test_split(df_credit, test_size=0.2, random_state=42, shuffle=True)
target = 'Class'

print("Logistic Reg : ")
X_train = train_df.drop([target], axis=1).values  # independant features
Y_train = train_df[target].values  # dependant variable
Y_test = test_df[target].values
X_test = test_df.drop([target], axis=1).values
b , W = logisticRegression(X_train,Y_train)

pred = lr_predict(X_train,b,W)
tn, fp, fn, tp = confusion_matrix(Y_train,pred).ravel()
# print(tn,fp,fn,tp)
print("\n<<<  For train data : >>>\n")
print("Accuracy : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

pred = lr_predict(X_test,b,W)
tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()
# print(tn,fp,fn,tp)
print("\n****For test data : ****\n")
print("Accuracy : ", ((tn+tp)*100.0)/(tn+fp+fn+tp))
print("True positive rate (sensitivity, recall, hit rate) : ",(tp*100.0)/(tp+fn))
print("True negative rate (specificity) : ", (tn*100.0)/(tn+fp))
print("Positive predictive value (precision) : ", (tp*100.0)/(tp+fp))
print("False discovery rate : ", (fp*100.0)/(fp+tp))
print("F1 score : ",(tp*200.0)/(2*tp + fp + fn ))

print("\n___________Adaboost__________\n")
k = [5,10,15,20]
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_train,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_train,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()

print("\nFor test data : \n")
for i in k:
    print("Number of boosting rounds :  ", i)
    h, z = adaboost(train_df,logisticRegression,target,i)
    X_test = test_df.drop([target], axis=1).values  # independant features
    Y_test = test_df[target].values  # dependant variabl
    ada_pred = adaboost_predict(X_test,h,z)
    tn, fp, fn, tp = confusion_matrix(Y_test,ada_pred).ravel()
    # print(tn,fp,fn,tp)
    print("Accuracy : ", (tp+tn)*100.0/(tn+fp+fn+tp))
    print()

