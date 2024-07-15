import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
df = pd.read_csv('train.csv')

# Basic data exploration
print(df.head())
print(df.info())
print(df.describe())

# Correlation matrix
correlation = df.corr()
plt.figure(figsize=(18, 15))
sns.heatmap(correlation, cmap='coolwarm', annot=True)
plt.show()

# Sort correlation with respect to 'price_range'
corr = df.corr()
corr.sort_values(['price_range'], ascending=False, inplace=True)
print(corr.price_range.head(21))

# Boxplot of 'ram' vs 'price_range'
sns.boxplot(x="price_range", y="ram", data=df)
plt.show()

# Pie chart for 'wifi'
labels = ["wifi-supported", 'Not supported']
values = df['wifi'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# Pie chart for 'price_range'
labels = ["0", '1', '2', '3']
values = df['price_range'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# Pie chart for 'four_g'
labels = ["4g-supported", 'Not supported']
values = df['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# Remove unwanted features
data = df.copy()
data.drop(['n_cores', 'm_dep', 'four_g', 'three_g', 'blue', 'clock_speed', 'sc_w', 'sc_h'], axis=1, inplace=True)
X = data.drop('price_range', axis=1)
Y = data['price_range']

# Random Forest function
def RF(X, Y, m_depth):
    compare_list_3 = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.23, random_state=0)
    for i in m_depth:
        clf = RandomForestClassifier(max_depth=i, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        compare_list_3.append((pd.Series({
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Train_Score": clf.score(X_train, y_train),
            "Test_Score": clf.score(X_test, y_test)
        }, name=i)))
    compare_list_3 = pd.DataFrame(compare_list_3).T
    return compare_list_3

compare_3 = RF(X, Y, np.arange(1, 30))
print('Accuracy of RF:', compare_3.loc['Accuracy',].max())
print(compare_3.idxmax(axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred2))

ConfusionMatrixDisplay.from_estimator(clf2, X_test, y_test, colorbar=False, cmap='Oranges_r')
plt.grid(False)
plt.show()

# Cross-validation
cv_scores = cross_val_score(clf2, X_train, y_train, cv=5)
mean_cv_score = cv_scores.mean() * 100
std_cv_score = cv_scores.std()
print("Mean Cross-Validation Accuracy:", mean_cv_score)
print("Cross-Validation Score Standard Deviation:", std_cv_score)

# Save the model
pickle.dump(clf2, open('model.pkl', 'wb'))
