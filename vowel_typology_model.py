# Exploring probabilistic vowel typology.
# (reproducing work by Ryan Cotterell and Jason Eisner:
#  https://arxiv.org/pdf/1705.01684.pdf)

# Adaption of the simple_experiments work into a packaged model to run
# probability functions on datasets.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
from itertools import combinations

import numpy as np
import tensorflow as tf
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

class VowelTypologyModel:

    # fields:
    #   data
    #   invcounts
    #   phi_norm_const
    #   zsc_data
    #   pca_data

    # --------------------------------------------------------------------------
    # consts

    #R = 16
    T = 0.5
    #AVG = 1951.138539042821
    #BATCH = 8
    #STEPS = 200
    #EPOCHS = 5
    BECKER_CORPUS_FILE = "../BeckerVowelCorpus.csv"

    # --------------------------------------------------------------------------
    # data loading and model initialization

    def __init__(self):
        # dict(key=(language * dialect), val=dict(key=phoneme, val=set(f1 * f2)))
        self.data = {}

        with open(self.BECKER_CORPUS_FILE) as corpus:
            reader = csv.DictReader(corpus)

            for row in reader:
                languagedialect = (row["Language"].strip(), row["Dialect"].strip())

                # track the current inventory
                inventory = {}
                if languagedialect in self.data:
                    inventory = self.data[languagedialect]

                for v in range(1, 14):
                    phonemeOS = row["V{}OS".format(v)].strip()
                    phonemePS = row["V{}PS".format(v)].strip()
                    if '(' in phonemeOS:
                        phonemeOS = phonemeOS[:phonemeOS.find('(')]
                    if '(' in phonemePS:
                        phonemePS = phonemePS[:phonemePS.find('(')]
                    f1 = row["V{}F1".format(v)].strip()
                    f2 = row["V{}F2".format(v)].strip()
                    family = row["Genetics"].strip()

                    phoneme = phonemePS if phonemePS else phonemeOS
                    if not phoneme:
                        continue

                    try:
                        float(f1)
                        float(f2)
                    except ValueError:
                        continue
                    if not float(f1) or not float(f2):
                        continue

                    phonemeEntries = set()
                    if phoneme in inventory:
                        phonemeEntries = inventory[phoneme]

                    phonemeEntries.add((float(f1), float(f2)))
                    inventory[phoneme] = phonemeEntries

                self.data[languagedialect] = inventory

        # counts of inventory occurence
        # dict(key=sorted phoneme inventory string, val=empirical count)
        self.invcounts = {}
        # used to normalize the total product of phi terms w.r.t. to inventory frequency
        self.phi_norm_const = float(0)

        for ld, inv in self.data.items():
            # sorted phoneme inventory string
            pset = "".join(sorted(list(inv.keys())))
            if pset not in self.invcounts:
                # the first time we see the set, add the focalization score to the phi_norm_const
                foc = [np.linalg.norm(np.array(samples.pop())) for samples in inv.values()]
                self.phi_norm_const += np.prod(np.array(foc))
                self.invcounts[pset] = 0.0
            self.invcounts[pset] += 1.0

        # ----------------------------------------------------------------------
        # simple embeddings

        f1s = []
        f2s = []
        ps  = []
        count = 0

        # tabulation contains 3-tuples of (start index, end index, inventory count)
        # this is used to recover per-inventory data points after working with all
        #   phonemes at once in the data processing steps (zscore and pca)
        tabulation = []

        all_vowels = set()
        for languagedialect, inventory in self.data.items():
            start = count
            for phoneme, samples in inventory.items():
                all_vowels.add(phoneme)
                if samples:
                    count += 1
                    f1, f2 = samples.pop()
                    f1s.append(f1)
                    f2s.append(f2)
                    ps.append(phoneme)
            end = count
            # sorted phoneme inventory string
            pset = "".join(sorted(list(inventory.keys())))
            invcount = self.invcounts[pset]
            # skip empty inventories!
            if start != end:
                indices = (start, end, invcount)
                tabulation.append(indices)

        print()
        print("Becker vowel corpus loaded!")
        #print(", ".join(sorted(list(all_vowels))))
        print(str(len(f1s)) + " phoneme samples")
        print("f1 ~ " + str(int(np.mean(f1s))) + " +/- " + str(int(np.std(f1s))))
        print("f2 ~ " + str(int(np.mean(f2s))) + " +/- " + str(int(np.std(f2s))))

        zsc_formant_data = list(zip(zip(zscore(f1s), zscore(f2s)), ps))
        self.zsc_data = np.array([(np.array(zsc_formant_data[i:j]),ic) for (i,j,ic) in tabulation])

        X = np.array(list(zip(f1s, f2s)))
        pca = PCA(n_components=2)
        pca.fit(X)
        pca_X = pca.transform(X)
        pca_formant_data = list(zip(pca_X, ps))
        self.pca_data = np.array([(np.array(pca_formant_data[i:j]),ic) for (i,j,ic) in tabulation])

        # *_data array's have form:
        # [ ( [((f1, f2), phoneme), ((f1, f2), phoneme), ...], inventory count ),
        #   ...
        # ]

        print()
        print("Probabilistic vowel typology model initalized!")
        print(str(len(self.data)) + " language dialects")
        print(str(len(self.invcounts)) + " unique vowel inventories")


    # --------------------------------------------------------------------------
    # probability functions

    #phonemeFreqs = {}
    #phonemeCount = 0.0
    #for phoneme in ps:
    #    if phoneme not in phonemeFreqs:
    #        phonemeFreqs[phoneme] = 0.0
    #    phonemeCount += 1.0
    #    phonemeFreqs[phoneme] += 1.0

    # phoneme probability function
    def phi(p):
        #p = p[1]
        #result = phonemeFreqs[p] / phonemeCount
        #if result < 0 or result > 1:
        #    print("[ERROR] phi probability of " + str(result))
        #return result
        result = np.linalg.norm(np.array(p[0]))
        #if result < 0 or result > 1:
        #    print("[ERROR] phi probability of " + str(result))
        return result

    # probability function for inter-phoneme interactions
    def psi(self, a, b, embeddings):
        x = embeddings(np.array(a[0]))
        y = embeddings(np.array(b[0]))
        # vector length of difference
        length = np.linalg.norm(x - y)
        # quasi-coulomb's law
        result = np.exp(np.negative(1 / (self.T * length)))
        #if result < 0 or result > 1:
        #    print("[ERROR] psi probability of " + str(result))
        return result

    # Bernoulli Point Process probability model
    def bpp(self, inv, embeddings):
        # pass the phoneme-annotated data points down to phi for debugging
        foc = [VowelTypologyModel.phi(p) for p in inv]
        result = np.prod(np.array(foc)) / self.phi_norm_const
        if result < 0 or result > 1:
            print("[ERROR] bpp probability of " + str(result))
        return result

    # Markov Point Process probability model
    def mpp(self, inv, embeddings):
        # pass the phoneme-annotated data points down to phi and psi for debugging
        foc = [VowelTypologyModel.phi(p) for p in inv]
        dis = [self.psi(p1, p2, embeddings) for (p1, p2) in combinations(inv, 2)]
        result = np.prod(np.append(np.array(foc), dis)) / self.phi_norm_const
        if result < 0 or result > 1:
            print("[ERROR] mpp probability of " + str(result))
        return result

    # --------------------------------------------------------------------------
    # experiments

    # cross entropy by taking the negative sum over N held out languages:
    #   - SUM (y' * log y)
    #   where y is predicted language inventory probability (according to mpp)
    #   and y' is empirical probability (number of inventory occurences / N)

    def cross_entropy(self, y, embeddings, model):
        #invcounts = {}
        #N = 0.0
        #for inv, ic in y:
        #    ps = []
        #    for fs, p in inv:
        #        ps.append(p)
        #    pset = "".join(sorted(ps))
        #    if pset not in invcounts:
        #        invcounts[pset] = 0.0
        #    invcounts[pset] += 1.0
        #    N += 1.0

        N = float(len(self.data))
        xent = 0.0

        for inv, ic in y:
            xent += np.log(model(inv, embeddings)) * ic / N
        return np.negative(xent)

    def cross_validate(self, data, embeddings, split_size=10):
        bestbppresult = float("inf")
        bestmppresult = float("inf")
        kf = KFold(n_splits=split_size)

        for i, (train_idx, val_idx) in enumerate(kf.split(data)):
            print("fold " + str(i+1) + "\tstarting at " + str(val_idx[0]) + "\twith cross-entropy = ", end='')
            val_x = data[val_idx]
            bppresult = self.cross_entropy(val_x, embeddings, self.bpp)
            mppresult = self.cross_entropy(val_x, embeddings, self.mpp)

            print(str(int(bppresult)) + " (bpp) / ", end='')
            print(str(int(mppresult)) + " (mpp)")

            if bppresult < bestbppresult:
                bestbppresult = bppresult
            if mppresult < bestmppresult:
                bestmppresult = mppresult
        return (bestbppresult, bestmppresult)

    def e(v):
        return v

    def test(self):
        print("\n---- Naive ZScore Embeddings ----")
        bppresult, mppresult = self.cross_validate(self.zsc_data, VowelTypologyModel.e)
        print("\tBPP achieved cross-entropy = " + str(bppresult))
        print("\tMPP achieved cross-entropy = " + str(mppresult))
        print("\n---- PCA Embeddings into R^2 ----")
        bppresult, mppresult = self.cross_validate(self.pca_data, VowelTypologyModel.e)
        print("\tBPP achieved cross-entropy = " + str(bppresult))
        print("\tMPP achieved cross-entropy = " + str(mppresult))

if __name__ == '__main__':
    model = VowelTypologyModel()
    model.test()
