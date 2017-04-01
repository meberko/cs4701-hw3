import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import sys

colorKey = {1: 'r', 0: 'b'}

class SVM:
    def __init__(self, tuples, ofname):
        self.tuples = tuples
        self.ofname = ofname
        self.x = []
        self.y = []
        self.colors = []

        # Gather all tuples and labels
        for t in self.tuples:
            self.x.append([1,t[0],t[1]])
            self.y.append(t[2])

        # Create color array
        for s in self.y:
            self.colors.append(colorKey[s])

    def plot(self):
        plt.figure()
        i=0
        for s in self.x:
            plt.plot(s[1],s[2],self.colors[i]+'o')
            i+=1
        plt.show()


if __name__ == '__main__':
    print 'Initializing Problem 3'

    # Check correct arguments
    if len(sys.argv) != 3:
        print 'usage: python problem3.py <input_filename> <output_filename>'
    else:
        print sys.argv[1]
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

        gd = SVM(tuples, sys.argv[2])
