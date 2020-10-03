import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, roc_auc_score
import statsmodels.api as sm

# Importing data

dataset = pd.read_csv("Credit Risk.csv")

# Dependent and Independent veriable selection

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Encode the target veriable, i.e. y veriable

LE_y = LabelEncoder()

y = LE_y.fit_transform(y)

# Now it's y veriable is in terms of 1s and 0s.

####################### 1st part is clear and now comes to the main part, i.e. Data clearning ###############

# Pre-processing

# Check for missing values and check for outliers.

nans = x.isnull().sum()
nans

# Learning more about perticular veriable. Let's say gender

# Gender

x['Gender'].describe()

x.Gender.value_counts()

x.Gender.isnull().sum()

# Filling NaNs 

x["Gender"].fillna("Male", inplace = True)

x.Gender.isnull().sum()


# Married

x['Married'].describe()

x.Married.value_counts()

x.Married.isnull().sum()

# Filling Nans

x["Married"].fillna("Yes", inplace = True)

x.Married.isnull().sum()


# Dependents

x["Dependents"].describe()

x.Dependents.value_counts()

x.Dependents.isnull().sum()

x["Dependents"].fillna("0", inplace = True)

x.Dependents.isnull().sum()


# Education

x["Education"].describe()

x.Education.isnull().sum()  # No missing values.



# Self_Employed

x["Self_Employed"].describe()

x.Self_Employed.isnull().sum()

x["Self_Employed"].fillna("No", inplace = True)

x.Self_Employed.isnull().sum()


# ApplicantIncome

x["ApplicantIncome"].describe()

x.ApplicantIncome.isnull().sum() # No missing values

# Check for outliers

x.boxplot("ApplicantIncome")

# Outliers treatment

q75, q25 = np.percentile(x.ApplicantIncome, [75, 25])

# Inter Quartile Range (iqr)

iqr = q75 - q25

iqr

# Upper Threshold value 

a = q75 + (1.5 * iqr)

a

# Lower Threshold value

b = q25 - (1.5 * iqr)

b

# So we will only consider the upper outliers and replace them with mean value

x.ApplicantIncome.loc[x["ApplicantIncome"] >= a] = np.mean(x['ApplicantIncome'])

x.ApplicantIncome.describe()

x.boxplot("ApplicantIncome")



# CoapplicantIncome

x["CoapplicantIncome"].describe()

x.CoapplicantIncome.isnull().sum() # No missing values

# Check for outliers

x.boxplot("CoapplicantIncome")

# Outliers treatment

q75C, q25C = np.percentile(x.CoapplicantIncome, [75, 25])

# Inter Quartile Range (iqr)

iqrC = q75C - q25C

iqrC

# Upper Threshold value 

aC = q75C + (1.5 * iqrC)

aC

# Lower Threshold value

bC = q25C - (1.5 * iqrC)

bC

# So we will only consider the upper outliers and replace them with mean value

x.CoapplicantIncome.loc[x["CoapplicantIncome"] >= a] = np.mean(x['CoapplicantIncome'])

x.CoapplicantIncome.describe()

x.boxplot("CoapplicantIncome")


# LoanAmount


x["LoanAmount"].describe()

x.LoanAmount.isnull().sum()

# 22 Missing values here

x["LoanAmount"].fillna(x["LoanAmount"].median(), inplace = True)

x.LoanAmount.isnull().sum()

# Check for outliers

x.boxplot("LoanAmount")

# Outliers treatment

q75L, q25L = np.percentile(x.LoanAmount, [75, 25])

# Inter Quartile Range (iqr)

iqrL = q75L - q25L

iqrL

# Upper Threshold value 

aL = q75L + (1.5 * iqrL)

aL

# Lower Threshold value

bL = q25L - (1.5 * iqrL)

bL

# So we will only consider the upper outliers and replace them with mean value

x.LoanAmount.loc[x["LoanAmount"] >= aL] = np.median(x['LoanAmount'])

x.LoanAmount.describe()

x.boxplot("LoanAmount")


# Loan_Amount_Term

x["Loan_Amount_Term"].describe()

x.Loan_Amount_Term.isnull().sum()

x.Loan_Amount_Term.unique()

x.Loan_Amount_Term.value_counts()

x["Loan_Amount_Term"].fillna(360.0, inplace = True)

x.Loan_Amount_Term.isnull().sum()


# Credit_History

x["Credit_History"].describe()

x.Credit_History.isnull().sum()

x.Credit_History.value_counts()

x["Credit_History"].fillna(1, inplace = True)

x.Credit_History.isnull().sum()


# Property_Area

x["Property_Area"].describe()

x.Property_Area.isnull().sum() # No missing values



##################### Done with the pre-processing part. Processed all independent veriables. ################

# Binning for Loan_Amount_Term.

x["Loan_Amount_Term_Bin"] = pd.cut(x["Loan_Amount_Term"], bins = [0, 120, 240, 360, 480], labels = ['0-120',
                             '120-240', '240-360', '360-480'])

x.Loan_Amount_Term_Bin.value_counts()


# Removing irrelavent veriables

x1 = x.drop(['Loan_ID', 'Loan_Amount_Term'], axis = 1)


# Making dummies

x2 = pd.get_dummies(x1, drop_first = True)


# Splitting dataset

x_train, x_test, y_train, y_test = tts(x2, y, test_size = 0.20, random_state = 0)


###############################################################################


# Model 

classifier = LR(random_state = 0)

classifier.fit(x_train, y_train)

# Training set prediction

y_pred_r = classifier.predict(x_train)

# Confusion matrix

cm_training = confusion_matrix(y_train, y_pred_r)

cm_training

# 80% correct prediction for traing set itself. 
# It's very bad


roc_auc_score(y_true = y_train, y_score = y_pred_r)

# AOC = 0.70

###############################################################################


# Prediction for test set

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

cm

# Model is performing 82% accurate.


# Advancing the model using backward elimination

classifier02 = sm.Logit(endog = y_train, exog = x_train).fit()

classifier02.summary()

# Prediction 

pred02 = classifier02.predict(x_test)

pred02 = (pred02 > 0.5).astype(int)

cm02 = confusion_matrix(y_test, pred02)

acc = (cm02[0][0] + cm02[1][1])/123

acc

# Now accuracy is 83.7% which is good.

roc_auc_score(y_true = y_test, y_score = pred02)

# Now ROC score is 0.725, which is better than earliar.


############################# Project Completed ###############################