import matplotlib.pyplot as plt
import numpy as np
import sys

class GradDec:
    def __init__(self, tuples, ofname):
        self.tuples = tuples
        self.ofname = ofname
        self.a = []
        self.w = []
        self.h = []
        self.asc = []
        self.wsc = []
        self.hsc = []

        # Gather all tuples and labels
        for t in self.tuples:
            self.a.append(t[0])
            self.w.append(t[1])
            self.h.append(t[2])

        mu_a = np.mean(self.a)
        mu_w = np.mean(self.w)
        mu_h = np.mean(self.h)
        std_a = np.std(self.a)
        std_w = np.std(self.w)
        std_h = np.std(self.h)

        for i in range(0,len(self.a)):
            self.asc.append((self.a[i]-mu_a)/std_a)
            self.wsc.append((self.w[i]-mu_w)/std_w)
            self.hsc.append((self.h[i]-mu_h)/std_h)
            print(('%f\t%f') % (self.a[i],self.asc[i]))

if __name__ == '__main__':
    print 'Initializing Problem 2'

    # Check correct arguments
    if len(sys.argv) != 3:
        print 'usage: python problem2.py <input_filename> <output_filename>'
    else:
        # Open given input file
        with open(sys.argv[1]) as inf:
            content = inf.readlines()
        with open(sys.argv[2],'w') as of:
            of.write("")

        # Clean up content and arrange into tuples
        content = [x.strip() for x in content]
        tuples = [x.split(',') for x in content]
        tuples = map(lambda t: [float(i) for i in t], tuples)

        gd = GradDec(tuples, sys.argv[2])
