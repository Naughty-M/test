import numpy as np
from AIAIndividual import AIAIndividual
import random
import copy
import matplotlib.pyplot as plt


class ArtificialImmuneAlgorithm:

    '''
    The class for artificial immune algorithm
    '''

    def __init__(self, sizepop, sizemem, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of [mutation rate, cloneNum]
        '''
        self.sizepop = sizepop
        self.sizemem = sizemem
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.clonePopulation = []
        self.memories = []
        self.cloneMemories = []
        self.popFitness = np.zeros(self.sizepop)
        self.popCloneFitness = np.zeros(
            int(self.sizepop * self.sizepop * params[1]))
        self.memfitness = np.zero(self.sizemem)
        self.memClonefitness = np.zero(
            int(self.sizemem * self.sizemem * params[1]))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = AIAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)
        for i in range(0, self.sizemem):
            ind = AIAIndividual(self.vardim, self.bound)
            ind.generate()
            self.memories.append(ind)

    def evaluatePopulation(self, flag):
        '''
        evaluation of the population fitnesses
        '''
        if flag == 1:
            for i in range(0, self.sizepop):
                self.population[i].calculateFitness()
                self.popFitness[i] = self.population[i].fitness
        else:
            for i in range(0, self.sizemem):
                self.memories[i].calculateFitness()
                self.memfitness[i] = self.memories[i].fitness

    def evaluateClone(self, flag):
        '''
        evaluation of the clone fitnesses
        '''
        if flag == 1:
            for i in range(0, self.sizepop):
                self.clonePopulation[i].calculateFitness()
                self.popCloneFitness[i] = self.clonePopulation[i].fitness
        else:
            for i in range(0, self.sizemem):
                self.cloneMemories[i].calculateFitness()
                self.memClonefitness[i] = self.cloneMemories[i].fitness

    def solve(self):
        '''
        evolution process of artificial immune algorithm
        '''
        self.t = 0
        self.initialize()
        self.best = AIAIndividual(self.vardim, self.bound)
        while (self.t < self.MAXGEN):
            # evolution of population
            self.cloneOperation(1)
            self.mutationOperation(1)
            self.evaluatePopulation(1)
            self.selectionOperation(1)

            # evolution of memories
            self.cloneOperation(2)
            self.mutationOperation(2)
            self.evaluatePopulation()
            self.selectionOperation(2)

            best = np.max(self.popFitness)
            bestIndex = np.argmax(self.popFitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.popFitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
            self.t += 1

        print("Optimal function value is: %f; " %
              self.trace[self.t - 1, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
        self.printResult()

    def cloneOperation(self, individuals):
        '''
        clone operation for alforithm immune algorithm
        '''
        newpop = []
        sizeInds = len(individuals)
        for i in range(0, sizeInds):
            for j in range(0, int(self.params[1] * sizeInds)):
                newpop.append(copy.deepcopy(individuals[i]))
        return newpop

    def selectionOperation(self, flag):
        '''
        selection operation for artificial immune algorithm
        '''
        if flag == 1:
            sortedIdx = np.argsort(-self.clonefit)
            for i in range(0, int(self.sizepop*self.sizepop*self.params[1]):
            tmpInd = individuals[sortedIdx[i]]
            if tmpInd.fitness > self.population[i].fitness:
                self.population[i] = tmpInd
                self.popFitness[i] = tmpInd.fitness
        else:
            pass
        newpop = []
        sizeInds = len(individuals)
        fitness = np.zeros(sizeInds)
        for i in range(0, sizeInds):
            fitness[i] = individuals[i].fitness
        sortedIdx = np.argsort(-fitness)
        for i in range(0, sizeInds):
            tmpInd = individuals[sortedIdx[i]]
            if tmpInd.fitness > self.population[i].fitness:
                self.population[i] = tmpInd
                self.popFitness[i] = tmpInd.fitness

    def mutationOperation(self, individuals):
        '''
        mutation operation for artificial immune algorithm
        '''
        newpop = []
        sizeInds = len(individuals)
        for i in range(0, sizeInds):
            newpop.append(copy.deepcopy(individuals[i]))
            r = random.random()
            if r < self.params[0]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                for k in range(0, self.vardim):
                    if newpop.chrom[mutatePos] < self.bound[0, mutatePos]:
                        newpop.chrom[mutatePos] = self.bound[0, mutatePos]
                    if newpop.chrom[mutatePos] > self.bound[1, mutatePos]:
                        newpop.chrom[mutatePos] = self.bound[1, mutatePos]
                newpop.calculateFitness()
        return newpop

    def printResult(self):
        '''
        plot the result of the artificial immune algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial immune algorithm for function optimization")
        plt.legend()
        plt.show()