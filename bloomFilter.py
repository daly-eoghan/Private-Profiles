# Code by PrasannallTM. Found at https://github.com/PrasannaIITM/Bloom-Filter

import mmh3 # murmurhash: is faster for blooms
import math
import numpy as np

class BloomFilter(object):

    def __init__(self, m, k):
        self.m = m # size of bloom filter
        self.k = k # number of hash functions
        self.n = 0 # total count of the elements inserted in the set
        self.bloomFilter = [0 for i in range(self.m)]
    
    def length(self):
        """
		returns the size of the filter -> was returning the number of items added was is fixed.
        """

        return self.m
        
    def _setAllBitsToZero(self):

        for i in range(len(self.bloomFilter)):
            self.bloomFilter[i] = 0
        self.n = 0
            
    def getBitArrayIndices(self, item):
        """
		hashes the key for k defined,
		returns a list of the indexes which have to be set
        """

        indexList = []
        for i in range(1, self.k + 1):
            indexList.append((hash(item) + i * mmh3.hash(item)) % self.m)
        return indexList
        
    def add(self, item):
        """
		Insert an item in the filter
        """
        
        for i in self.getBitArrayIndices(item):
            self.bloomFilter[i] = 1
        
        self.n += 1
    
    def contains(self, key):
        """
		returns whether item exists in the set or not
        """

        for i in self.getBitArrayIndices(key):
            if self.bloomFilter[i] != 1:
                return False
        return True

    def calculateFalsePositive(self):
        """
		Calculates the statistics of a filter
		Probability of False Positives, predicted false positive rate, n, m, k.
        """
        
        n = float(self.n)
        m = float(self.m)
        k = float(self.k)
        probability_fp = math.pow((1.0 - math.exp(-(k*n)/m)), k)

        return probability_fp
        
    def generateStats(self):
        """
		Prints out the no. of elements added, size of the bloom filter,
        no. of hash functions used, the probability of false positives and
        the rate of such.
        """

        probability_fp = self.calculateFalsePositive()

        print("Number of elements entered in filter: ", self.n)
        print("Number of bits in filter: ", self.m)
        print("Number of hash functions used: ", self.k)

        print("Predicted Probability of false positives: ", probability_fp)
        print("Predicted false positive rate: ", probability_fp * 100.0, "%")
        
    def reset(self):
        """
		Resets the filter and clears old values and statistics
        """

        self._setAllBitsToZero()
 
    def list_to_array(self):
        """
		Converts the underlying structure from a list into a numpy array.
        """

        self.bloomFilter = np.array(self.bloomFilter)