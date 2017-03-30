import matplotlib.pyplot as plt
import numpy as np
import sys

colorKey = {1: 'r', -1: 'b'}

class Perceptron:
    def __init__(self, tuples, ofname):
        self.tuples = tuples
        self.ofname = ofname
        self.weights = [0,0]
        self.x = []
        self.labels = []
        self.gamma = 0.01

        # Gather all tuples and labels
        for t in self.tuples:
            self.x.append(np.array([1, t[0], t[1]]))
            self.labels.append(t[2])
        self.colors = map(lambda i: colorKey[i], self.labels)

        # Start by randomizing weights
        self.weights = np.array([2*np.random.ranf()-1, 2*np.random.ranf()-1, 2*np.random.ranf()-1])

    def setTuples(self, tuples):
        self.tuples = tuples

    def getWrong(self):
        incorrect = []
        for i in range(0,len(self.x)):
            y = np.dot(self.weights, self.x[i])
            predicted = 1 if y > 0 else -1
            if predicted != self.labels[i]:
                incorrect.append([self.labels[i],self.x[i]])
        return incorrect

    def learn(self):
        iters = 0

        # While there are still misclassified points
        while len(self.getWrong())!= 0:
            iters += 1
            # Write w1, w2, b to output file
            with open(self.ofname,'a') as of:
                of.write(('%f, %f, %f') % (self.weights[1], self.weights[2], self.weights[0]) + '\n')

            # Get the wrong points, pick a random one, and
            wrong = self.getWrong()
            rand_wrong = wrong[np.random.randint(0,len(wrong))]
            label = rand_wrong[0]
            x = rand_wrong[1]
            y = np.dot(self.weights, x)
            predicted = 1 if y > 0 else -1
            self.weights += self.gamma*label*x
        self.plot(iters)

    def plot(self, its):
        plt.figure()
        for i in range(0,len(self.x)):
            plt.plot(self.x[i][1], self.x[i][2], self.colors[i]+'o')
        l = np.linspace(0,16)
        a = -self.weights[1]/self.weights[2]
        b = -self.weights[0]/self.weights[2]
        plt.plot(l, a*l+b)
        plt.title(('%d Iterations') % (its))
        plt.axis([0,16,-30,30])
        plt.show()

if __name__ == '__main__':
    print 'Initializing Percy the Perceptron'

    # Check correct arguments
    if len(sys.argv) != 3:
        print 'usage: python problem1.py <input_filename> <output_filename>'
    else:
        # Open given input file
        with open(sys.argv[1]) as inf:
            content = inf.readlines()
        with open(sys.argv[2],'w') as of:
            of.write("")

        # Clean up content and arrange into tuples
        content = [x.strip() for x in content]
        tuples = [x.split(',') for x in content]
        tuples = map(lambda t: [int(i) for i in t], tuples)

        # Create Percy the Perceptron and tell him to learn on the tuples
        percy = Perceptron(tuples, sys.argv[2])
        percy.learn()

