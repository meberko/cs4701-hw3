import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sys

colorKey = {1: 'r', 0: 'b'}

printKey = {'linear':'svm_linear', 'poly':'svm_polynomial', 'rbf':'svm_rbf', 'log':'logistic', 'knn':'knn', 'dt':'decision_tree', 'rf':'random_forest'}

class SVM:
    def __init__(self, tuples, ofname, kernel, C=[0], gamma=['auto'], degree=[3], nn=[0], leaf_size=[0], max_depth=[0], min_samples_split=[0]):
        self.tuples = tuples
        self.ofname = ofname
        self.x1 = []
        self.x2 = []
        self.x = []
        self.y = []
        self.xsc = []
        self.colors = []
        self.svm=''
        if kernel in ['linear', 'poly', 'rbf']:
            self.svm = SVC(kernel=kernel)
            params = {'C':C, 'gamma':gamma, 'degree':degree}
        elif kernel=='log':
            self.svm = LogisticRegression()
            params = {'C':C}
        elif kernel=='knn':
            self.svm = KNeighborsClassifier()
            params = {'n_neighbors':nn, 'leaf_size':leaf_size}
        elif kernel=='dt':
            self.svm = DecisionTreeClassifier()
            params = {'max_depth':max_depth, 'min_samples_split':min_samples_split}
        elif kernel=='rf':
            self.svm = RandomForestClassifier()
            params = {'max_depth':max_depth, 'min_samples_split':min_samples_split}
        self.gs = GridSearchCV(self.svm, params, cv=5)

        # Gather all tuples and labels
        for t in self.tuples:
            self.x.append([t[0],t[1]])
            self.y.append(t[2])
            self.x1.append(t[0])
            self.x2.append(t[1])

        # Gather statistics on features
        x1_mean = np.mean(self.x1)
        x1_std = np.std(self.x1)
        x2_mean = np.mean(self.x2)
        x2_std = np.std(self.x2)

        # Scale x1 and x2
        for i in range(0,len(self.x)):
            self.xsc.append([(self.x1[i]-x1_mean)/x1_std , (self.x2[i] - x2_mean)/x2_std])

        # Create color array
        for s in self.y:
            self.colors.append(colorKey[s])

        # Split data into train/test sets
        self.xsc_train, self.xsc_test, self.y_train, self.y_test = train_test_split(self.xsc, self.y, test_size=0.4)

        # Fit grid search on train data
        self.gs.fit(self.xsc_train, self.y_train)
        with open(self.ofname, 'a') as of:
            of.write(('%s, %f, %f\n') % (printKey[kernel], self.gs.best_score_, self.gs.score(self.xsc_test, self.y_test)))

    def plot(self):
        plt.figure()
        i=0
        for s in self.xsc:
            plt.plot(s[0],s[1],self.colors[i]+'o')
            i+=1
        plt.show()


if __name__ == '__main__':
    #print 'Initializing Problem 3'

    # Check correct arguments
    if len(sys.argv) != 3:
        print 'usage: python problem3.py <input_filename> <output_filename>'
    else:
        # Open given input file
        with open(sys.argv[1]) as inf:
            content = inf.readlines()
        with open(sys.argv[2],'w') as of:
            of.write("")

        # Clean up content and arrange into tuples
        content = [x.strip() for x in content]
        content.pop(0)
        tuples = [x.split(',') for x in content]
        tuples = map(lambda t: [float(i) for i in t], tuples)

        # Run analysis on SVMs with linear kernel
        kern = 'linear'
        #print 'SVM Linear Kernel'
        s = SVM(tuples, sys.argv[2], kern, C=[0.1,0.5,1,5,10,50,100])

        # Run analysis on SVMs with polynomial kernel
        kern = 'poly'
        #print 'SVM Polynomial Kernel'
        s = SVM(tuples, sys.argv[2], kern, C=[0.1,1,3], gamma=[0.1,1], degree=[4,5,6])

        # Run analysis on SVMs with rbf kernel
        kern = 'rbf'
        #print 'SVM RBF Kernel'
        s = SVM(tuples, sys.argv[2], kern, C=[0.1,0.5,1,5,10,50,100], gamma=[0.1,0.5,1,3,6,10])

        # Run analysis on logistic regression
        kern = 'log'
        #print 'Logistic Regression'
        s = SVM(tuples, sys.argv[2], kern, C=[0.1,0.5,1,5,10,50,100])

        # Run analysis on K nearest neighbors
        kern = 'knn'
        #print 'K Nearest Neighbors'
        s = SVM(tuples, sys.argv[2], kern, nn=range(1,51), leaf_size=map(lambda x:5*x, range(1,13)))

        # Run analysis on decision trees
        kern = 'dt'
        #print 'Decision Trees'
        s = SVM(tuples, sys.argv[2], kern, max_depth=range(1,51), min_samples_split=range(2,11))

        # Run analysis on random forest
        kern = 'rf'
        #print 'Random Forest'
        s = SVM(tuples, sys.argv[2], kern, max_depth=range(1,51), min_samples_split=range(2,11))

