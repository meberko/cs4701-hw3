import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

class GradDec:
    def __init__(self, tuples, ofname, alpha, iters):
        self.tuples = tuples
        self.ofname = ofname
        self.iters = iters
        self.alpha = alpha

        # Configurational stuff
        self.R = 0
        self.delta = float('inf')
        self.beta = np.array([0,0,0])

        # Data vectors, labels
        self.x = []
        self.y = []
        self.n = 0

        # Individual attributes and scaled attributes
        self.a = []
        self.w = []
        self.h = []
        self.asc = []
        self.wsc = []

        # Gather all ages, weights, and labels (heights)
        for t in self.tuples:
            self.a.append(t[0])
            self.w.append(t[1])
            self.h.append(t[2])

        # Get stats (mean & stdev)
        mu_a = np.mean(self.a)
        mu_w = np.mean(self.w)
        mu_h = np.mean(self.h)
        std_a = np.std(self.a)
        std_w = np.std(self.w)
        std_h = np.std(self.h)

        # Scale
        for i in range(0,len(self.a)):
            self.asc.append((self.a[i]-mu_a)/std_a)
            self.wsc.append((self.w[i]-mu_w)/std_w)

        # Gather x vectors
        for i in range(0,len(self.asc)):
            self.x.append(np.array([1,self.asc[i],self.wsc[i]]))
            self.y.append(self.h[i])

        self.n = len(self.x)
        R = 0
        for i in range(0,len(self.x)):
            R += (self.y[i] - np.dot(self.beta,self.x[i]))**2
        self.R = R/self.n

    def learn(self):
        for i in range(0,self.iters):
            prevR = self.R
            for i in range(0,self.n):
                self.beta = self.beta + (self.alpha/float(self.n))*(self.y[i]-np.dot(self.beta,self.x[i]))*self.x[i]
            self.updateR()
            self.delta = abs(self.R-prevR)

    def updateR(self):
        R = 0
        for i in range(0,self.n):
            R += (self.y[i] - np.dot(self.beta,self.x[i]))**2
        self.R = R/self.n

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        lx, ly = np.meshgrid([-3,3], [-3,3])
        ax.plot(self.asc,self.wsc,self.y, 'ro')
        ax.plot_surface(lx,ly,self.beta[0]+self.beta[1]*lx+self.beta[2]*ly, color='b')
        ax.set_title(('Alpha = %f') % (self.alpha))
        ax.set_xlabel('Age (Years)')
        ax.set_ylabel('Weight (Kilograms)')
        ax.set_zlabel('Height (Meters)')
        plt.show()

    def printResult(self):
        with open(self.ofname,'a') as of:
            if self.alpha < 0.01:
                of.write((('%.3f,%d,%.4f,%.4f,%.4f') % (self.alpha, self.iters, self.beta[0], self.beta[1], self.beta[2])+'\n'))
            elif self.alpha < 0.1:
                of.write((('%.2f,%d,%.4f,%.4f,%.4f') % (self.alpha, self.iters, self.beta[0], self.beta[1], self.beta[2])+'\n'))
            elif self.alpha < 1:
                of.write((('%.1f,%d,%.4f,%.4f,%.4f') % (self.alpha, self.iters, self.beta[0], self.beta[1], self.beta[2])+'\n'))
            else:
                of.write((('%d,%d,%.4f,%.4f,%.4f') % (self.alpha, self.iters, self.beta[0], self.beta[1], self.beta[2])+'\n'))

if __name__ == '__main__':
    #print 'Initializing Problem 2'

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

        for a in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            gd = GradDec(tuples, sys.argv[2], a, 100)
            gd.learn()
            gd.printResult()
        gd = GradDec(tuples, sys.argv[2], 7.15, 16)
        gd.learn()
        gd.printResult()
