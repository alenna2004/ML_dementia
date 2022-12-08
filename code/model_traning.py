"""
This code is used for parsing image data to table data and training four ML models(decicion tree, SVM, Random Forest and multilayer perceptron).
"""
import csv
import cv2
import os
import pandas as pd
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#Here you have to place the path to folder with images and the lists of names of images of different typef of dementia.
PATH = ''
im_names_0 = [...]
im_names_1 = [...]
im_names_2 = [...]


#dementia typy codes: 0 not demented, 1 alzheimer dementia, 2 for frontotemporal dementia


def generate_csv_table(im_names_data, path, dem_type, table_name):
    """
    This function parses image to a 1D array and adds this array as a row in a table. In the last cell of the row it places dementia type.
    """
    dict_list = []
    fieldnames = []
    for i in range(208):
        for j in range(176):
            fieldnames.append(f'{str(i)}_{str(j)}')
    fieldnames += ['dementia_class']
    for name in im_names_data:
        img = cv2.imread(f'{path}/{name}',0)
        row = {}
        row['dementia_class'] = dem_type
        for i in range(208):
            for j in range(176):
                row[f'{str(i)}_{str(j)}'] = int(img[i][j])
        dict_list.append(row)
    with open(table_name + '.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(table_name + '.csv').st_size == 0:
            writer.writeheader()
        writer.writerows(dict_list)

#Here we parse images data to table called dementia_data.
generate_csv_table(im_names_0, PATH', 0 , 'dementia_data')
generate_csv_table(im_names_1, PATH', 1 , 'dementia_data')
generate_csv_table(im_names_2, PATH', 2 , 'dementia_data')


def count_auc(y_test, predicted):
    """
    This function counts AUC of trained model comparing predictions and y_test data.
    """
    right_predictions = 0
    for i in range(len(y_test)):
        if y_test[i] == predicted[i]:
            right_predictions +=1
    print(right_predictions, len(y_test), right_predictions/len(y_test))


#Here we split data for test and train samples and training four models.
df = pd.read_csv('dementia_data.csv')
y = df.dementia_class.values
x = df.drop('dementia_class', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(x_train, y_train)
clf2 = svm.SVC(random_state=42)
clf2 = clf2.fit(x_train, y_train)
clf3 = RandomForestClassifier(n_estimators=120, random_state=42)
clf3 = clf3.fit(x_train, y_train)
clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=42)
clf4.fit(x_train, y_train)
#Here we make predictions using trained models and count AUC.
predicted = clf.predict(x_test)
predicted2 = clf2.predict(x_test)
predicted3 = clf3.predict(x_test)
predicted4 = clf3.predict(x_test)
auc1 = count_auc(y_test, predicted)
auc2 = count_auc(y_test, predicted2)
auc3 = count_auc(y_test, predicted3)
auc4 = count_auc(y_test, predicted4)
