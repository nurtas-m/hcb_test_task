import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import cycle
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy import interp
import copy
import statistics
from IPython.core.debugger import Tracer


class DataSet(object):

    def __init__(self, path):
        raw_lines = open(path).readlines()
        self.labels = raw_lines[0].strip().split(",")
        self.raw_dataset = []
        for line in raw_lines[1:]:
            line_split = line.strip().split(",")
            self.raw_dataset.append(line_split)
        dic = self.__create_customerid_pk_dic(self.raw_dataset)
        table = self.__create_customerid_pk_table(dic)
        self.__count_unique_values(table)

    def __create_customerid_pk_dic(self, table):
        # Create dictionary with customer_id keys
        dic, self.__customerid_callingnum_map = {}, {}
        table_id = 0
        for line_id, line in enumerate(table):
            if int(line[5]) not in self.__customerid_callingnum_map.keys():
                if int(line[27]) != 2015:
                    raise ValueError("Check year in line {0}".format(line_id))
                if (int(line[5])-1) != table_id:
                    raise ValueError("Customerid in table not ordered. Check line {0}".format(line_id))
                table_id += 1
                self.__customerid_callingnum_map[int(line[5])] = int(line[4])
                line_add = copy.deepcopy(line[:4])
                line_add.extend(copy.deepcopy(line[6:24]))
                line_add.append(int(line[26]))
                dic[int(line[5])] = {"const_data":line_add}
            else:
                curr_data = copy.deepcopy(line[:4])
                curr_data.extend(copy.deepcopy(line[6:24]))
                curr_data.append(int(line[26]))
                if self.__constant_data_changed(dic[int(line[5])]["const_data"], curr_data):
                    raise ValueError("Constant data of customer {0} changed".format(line[5]))

            if int(line[28]) == 1:
                dic[int(line[5])]["totalcallduration1"] = int(line[24])
                dic[int(line[5])]["avgcallduration1"] = int(line[25])
            elif int(line[28]) == 2:
                dic[int(line[5])]["totalcallduration2"] = int(line[24])
                dic[int(line[5])]["avgcallduration2"] = int(line[25])
            else:
                dic[int(line[5])]["totalcallduration3"] = int(line[24])
                dic[int(line[5])]["avgcallduration3"] = int(line[25])
            
        return dic

    def __create_customerid_pk_table(self, dic):
        # Modify initial table to the form where one row belongs to one customer
        table = []
        for table_id in range(len(dic)):
            line = dic[table_id+1]["const_data"]
            if "totalcallduration1" in dic[table_id+1].keys():
                line.append(dic[table_id+1]["totalcallduration1"])
                line.append(dic[table_id+1]["avgcallduration1"])
            else:
                line.append(0)
                line.append(0)
            if "totalcallduration2" in dic[table_id+1].keys():
                line.append(dic[table_id+1]["totalcallduration2"])
                line.append(dic[table_id+1]["avgcallduration2"])
            else:
                line.append(0)
                line.append(0)
            if "totalcallduration3" in dic[table_id+1].keys():
                line.append(dic[table_id+1]["totalcallduration3"])
                line.append(dic[table_id+1]["avgcallduration3"])
            else:
                line.append(0)
                line.append(0)
            table.append(line)
        return table

    def __constant_data_changed(self, prev_data, curr_data):
        # Check if the data that supposed to be constant are different
        for val_id, val in enumerate(prev_data):
            if val != curr_data[val_id]:
                return True
        return False

    def __mark_columns_type(self):
        self.columns_type = collections.OrderedDict()
        self.columns_type["age"] = "num"
        self.columns_type["annualincome"] = "num"
        self.columns_type["callproprate"] = "float"
        self.columns_type["callfailurerate"] = "float"
        self.columns_type["customersuspended"] = "bin"
        self.columns_type["education"] = "enum"
        self.columns_type["gender"] = "bin"
        self.columns_type["homeowner"] = "bin"
        self.columns_type["maritalstatus"] = "bin"
        self.columns_type["monthlybilledamount"] = "num"
        self.columns_type["noadditionallines"] = "bin"
        self.columns_type["numberofcomplaints"] = "num"
        self.columns_type["numberofmonthunpaid"] = "num"
        self.columns_type["numdayscontractequipmentplanexpiring"] = "num"
        self.columns_type["occupation"] = "enum"
        self.columns_type["penaltytoswitch"] = "num"
        self.columns_type["state"] = "enum"
        self.columns_type["totalminsusedinlastmonth"] = "num"
        self.columns_type["unpaidbalance"] = "num"
        self.columns_type["usesinternetservice"] = "bin"
        self.columns_type["usesvoiceservice"] = "bin"
        self.columns_type["percentagecalloutsidenetwork"] = "float"
        self.columns_type["churn"] = "num"
        self.columns_type["totalcallduration1"] = "num"
        self.columns_type["avgcallduration1"] = "num"
        self.columns_type["totalcallduration2"] = "num"
        self.columns_type["avgcallduration2"] = "num"
        self.columns_type["totalcallduration3"] = "num"
        self.columns_type["avgcallduration3"] = "num"


    def __change_data_types(self, table):
        # Transform data types in table
        table_type_changed = []
        for line in table:
            line_type_changed = []
            for label_id, (label, label_type) in enumerate(self.columns_type.items()):
                if label_type == "num":
                    line_type_changed.append(int(line[label_id]))
                elif label_type == "float":
                    line_type_changed.append(float(line[label_id]))
                elif label_type == "bin":
                    if line[label_id] in ("Yes", "Male", "Married"):
                        line_type_changed.append(1)#True)
                    else:
                        line_type_changed.append(0)#False)
                else:
                    # Enum columns
                    if label_type != "state":
                        if line[label_id] == "PhD or equivalent":
                            line_type_changed.append(3)
                        elif line[label_id] in ("Bachelor or equivalent", "Non-technology Related Job"):
                            line_type_changed.append(1)
                        elif line[label_id] in ("Master or equivalent", "Technology Related Job"):
                            line_type_changed.append(2)
                        else:
                            line_type_changed.append(0)
                    else:
                        # Range states by their economic condition
                        # West Coast
                        if line[label_id] in ("CA", "WA", "OR", "HI"):
                            line_type_changed.append(7)
                        # New England
                        elif line[label_id] in ("ME", "NH", "VT", "MA", "RI", "CT"):
                            line_type_changed.append(6)
                        # Mountain
                        elif line[label_id] in ("AK", "ID", "MT", "WY", "UT", "CO", "ND", "SD", "MN", "NE", "WI", "IA"):
                            line_type_changed.append(5)
                        # Mid-Atlantic
                        elif line[label_id] in ("NY", "NJ", "PA", "DE", "MD"):
                            line_type_changed.append(4)
                        # North Central
                        elif line[label_id] in ("OH", "MI", "IN", "IL", "MO", "KS"):
                            line_type_changed.append(3)
                        # West South
                        elif line[label_id] in ("NV", "AZ", "NM", "TX", "OK"):
                            line_type_changed.append(2)
                        # South Atlantic
                        elif line[label_id] in ("VA", "NC", "SC", "GA", "FL"):
                            line_type_changed.append(1)
                        # East South
                        elif line[label_id] in ("LA", "AR", "MS", "AL", "TN", "KY", "WV"):
                            line_type_changed.append(0)
                        else:
                            line_type_changed.append(0)
            table_type_changed.append(copy.deepcopy(line_type_changed))
        return table_type_changed

    def __find_max_mode(self, list1):
        list_table = statistics._counts(list1)
        len_table = len(list_table)

        if len_table == 1:
            max_mode = statistics.mode(list1)
        else:
            new_list = []
            for i in range(len_table):
                new_list.append(list_table[i][0])
            max_mode = max(new_list) # use the max value here
        return max_mode

    def __count_unique_values(self, table):
        # Count statistics by columns
        self.__mark_columns_type()
        table = self.__change_data_types(table)
        self.dataframe = pd.DataFrame(table, columns = self.columns_type.keys())
        
        self.unique_values = {}
        for label_id, label in enumerate(self.columns_type.keys()):
            if self.columns_type[label] in ["bin", "enum"]:
                self.unique_values[label] = collections.Counter()
                for val in self.dataframe[label]:
                    self.unique_values[label][val] += 1
            elif self.columns_type[label] in ["num", "float"]:
                self.unique_values[label] = {}
                self.unique_values[label]["min"] = min(self.dataframe[label])
                self.unique_values[label]["max"] = max(self.dataframe[label])
                self.unique_values[label]["mean"] = statistics.mean(self.dataframe[label])
                self.unique_values[label]["median"] = statistics.median(self.dataframe[label])
                self.unique_values[label]["mode"] = self.__find_max_mode(self.dataframe[label])#statistics.mode(dataframe[label])

    def train_test_split(self):
         # Split dataset to train and test sets
        trainset = self.dataframe.loc[:round(len(self.dataframe)*0.9)]
        testset = self.dataframe.loc[round(len(self.dataframe)*0.9):]
        self.X_train = trainset.loc[ : , trainset.columns != "churn"]
        self.y_train = trainset.loc[ : , "churn"]
        self.X_test = testset.loc[ : , testset.columns != "churn"]
        self.y_test = testset.loc[ : , "churn"]
                

ds = DataSet('telco-customer-churn.csv')
ds.train_test_split()

X_train, X_test, y_train, y_test = np.array(ds.X_train), np.array(ds.X_test), np.array(ds.y_train), np.array(ds.y_test)

h = .02  # step size in the mesh

names = [#"Nearest Neighbors"]#,
         #"Linear SVM",
         #"RBF SVM",
         #"Gaussian Process",
         #"Decision Tree",
         #"Random Forest",
         #"Neural Net"#,
         #"AdaBoost",
         #"Naive Bayes",
         #"QDA"]
         "Logistic Regression"
         ]

classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(hidden_layer_sizes=(1000, 200, 50),
    #              activation="relu",
    #              solver="adam",
    #              alpha=0.1,
    #              max_iter=1000,
    #              learning_rate="invscaling",
    #              learning_rate_init=0.01#,
    #              )
    #AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()]
    LogisticRegression(#random_state=0, #solver='sag',
                       class_weight='balanced')#,
                       #multi_class='multinomial')
    ]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)

r_num = 0
w_num = 0
stats = {}
for name, clf in zip(names, classifiers):
    stats[name] = {"tp":0, "tn":0, "fp":0, "fn":0}
    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i] == 1:
            stats[name]["tp"] += 1
        elif y_pred[i] == y_test[i]:
            stats[name]["tn"] += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 1:
            stats[name]["fn"] += 1
        else:
            stats[name]["fp"] += 1
    acc = (stats[name]["tp"] + stats[name]["tn"]) / len(y_test)
    if (stats[name]["tp"] + stats[name]["fp"]) > 0:
        prec = stats[name]["tp"] / (stats[name]["tp"] + stats[name]["fp"])
    else:
        prec = 0
    if (stats[name]["tp"] + stats[name]["fn"]) > 0:
        recall = stats[name]["tp"] / (stats[name]["tp"] + stats[name]["fn"])
    else:
        recall = 0
    if (prec + recall) > 0:
        f1_score = 2 * prec * recall / (prec + recall)
    else:
        f1_score = 0
    print(name+' Accuracy: ' + str(round(acc * 100, 1)) + '%')
    print(name+' Precision: ' + str(round(prec * 100, 1)) + '%')
    print(name+' Recall: ' + str(round(recall * 100, 1)) + '%')
    print(name+' F1_Score: ' + str(round(f1_score * 100, 1)) + '%')
    score = clf.score(X_test, y_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
#for i in range(2):
fpr[1], tpr[1], _ = roc_curve(y_test, y_score)
roc_auc[1] = auc(fpr[1], tpr[1])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
