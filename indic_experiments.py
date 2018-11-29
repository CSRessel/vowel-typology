
from functools import reduce
from itertools import combinations
import random

import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import KFold

from vowel_typology_model import VowelTypologyModel
from indic_formant_data import IndicFormantData

K = 10
CLOZE_TRIALS = 100

model = VowelTypologyModel()
formant_data = IndicFormantData()

# currently unnecessary aggregated data
# ----
#aggr_data, aggr_labs = formant_data.aggr
#
## TODO: hardcoded phoneme label conversions?
##fest2ipa = dict()
#
## TODO: remove uncommon or exceptional vowels from dataset?
#unknown = set()
#
## remove samples with labels unknown w.r.t. the becker vowel corpus annotations
## (backwards so we don't offset indices)
#for i, (sample, label) in enumerate(zip(reversed(aggr_data), reversed(aggr_labs))):
#    i = formant_data.N - 1 - i
#    if label in unknown:
#        aggr_data = np.delete(aggr_data, i, axis=0)
#        aggr_labs = np.delete(aggr_labs, i)
#
## delete the third value (f3) along axis=2
#aggr_data = np.delete(aggr_data, 2, axis=2)
#zsc_data = zscore(aggr_data, axis=1)
#
## reproducable results
random.seed(1300031)
np.random.seed(1300031)
#np.random.shuffle(zsc_data)

def e(v):
    return v

# average per-vowel measurements (using middle four intervals), returning an
# inventory with one entry per vowel
def preprocess(samples, labels):
    f1s = {}
    f2s = {}
    # average per-vowel measurements (using middle four intervals)
    for sample, vowel in zip(val_x, val_y):
        for interval in sample[3:7]:
            f1sum = 0.0
            f1count = 0
            f2sum = 0.0
            f2count = 0
            if vowel in f1s:
                assert vowel in f2s
                f1sum, f1count = f1s[vowel]
                f2sum, f2count = f2s[vowel]

            f1sum += interval[0]
            f1count += 1
            f2sum += interval[1]
            f2count += 1
            f1s[vowel] = (f1sum, f1count)
            f2s[vowel] = (f2sum, f2count)

    vowel_inv = []
    for v in f1s.keys(): #(v1, (f1sum, f1count)), (v2, (f2sum, f2count)) in zip(f1s.items(), f2s.items()):
        assert v in f2s
        f1sum, f1count = f1s[v]
        f2sum, f2count = f2s[v]
        vowel_inv.append((np.array([f1sum / f1count, f2sum / f2count]), v))

    return vowel_inv

# voice (aggr, tel_ss_tel, tam_sdr_artic, etc)
#     task (10-fold probability)
#         bpp result (probability or accuracy)
#             mpp result (probability or accuracy)
all_results = []

# ------------------------------------------------------------------------------
# 10 fold evaluation of probability of the fold's validation set's assigned
# probability mass

kf = KFold(n_splits=K)

for voice, (voice_labs, voice_data) in formant_data.data.items():

    # TODO: hardcoded phoneme label conversions?
    #fest2ipa = dict()

    # TODO: remove uncommon or exceptional vowels from dataset?
    unknown = set()

    # remove samples with labels unknown w.r.t. the becker vowel corpus annotations
    # (backwards so we don't offset indices)
    for i, (sample, label) in enumerate(zip(reversed(voice_data), reversed(voice_labs))):
        i = formant_data.N - 1 - i
        if label in unknown:
            voice_data = np.delete(voice_data, i, axis=0)
            voice_labs = np.delete(voice_labs, i)

    # delete the third value (f3) along axis=2
    voice_data = np.delete(voice_data, 2, axis=2)
    zsc_data = zscore(voice_data, axis=1)

    rng_state = np.random.get_state()
    np.random.shuffle(zsc_data)
    np.random.shuffle(voice_labs)

    bestbppresult = float("inf")
    bestmppresult = float("inf")
    print()
    print(f'Running {K} folds of estimating probability mass for {voice}\'s inventory')

    for i, (train_idx, val_idx) in enumerate(kf.split(zsc_data, voice_labs)):
        print(f'fold {str(i+1)}\tstarting at {str(val_idx[0])}\twith probability = ', end='')
        val_x = zsc_data[val_idx]
        val_y = voice_labs[val_idx]
        vowel_inv = preprocess(val_x, val_y)

        bppresult = model.bpp(inv=vowel_inv, embeddings=e)
        mppresult = model.mpp(inv=vowel_inv, embeddings=e)
        print(f'{str(int(bppresult))} (bpp) / ', end='')
        print(f'{str(int(mppresult))} (mpp)')
        if bppresult < bestbppresult:
            bestbppresult = bppresult
        if mppresult < bestmppresult:
            bestmppresult = mppresult

    # NOTES
    # essentially zero probability, because
    #   A) the model doesn't do focalization correctly
    #   B) these vowel inventories are real big, even in the KFold
    # need to fix the VowelTypologyModel, and consider removing the least common
    # vowels from this dataset?

    print(f'\tBPP assigns probability = {str(bestbppresult)}')
    print(f'\tMPP assigns probability = {str(bestmppresult)}')
    all_results.append((voice, "10-fold probability", bestbppresult, bestmppresult))

# ------------------------------------------------------------------------------
# CLOZE_TRIALS for each of: cloze task on 1 vowel removed, 0 or 1 vowels
# removed, and 0 or 1 or 2 vowels removed (all through model inference)

for voice, (voice_labs, voice_data) in formant_data.data.items():

    total_vowel_inv = preprocess(voice_data, voice_labs)
    inv = {}
    for fs, v in total_vowel_inv:
        inv[v] = fs
    bpp1s = []
    mpp1s = []
    bpp01s = []
    mpp01s = []
    bpp012s = []
    mpp012s = []

    print()
    print(f'Running {CLOZE_TRIALS} trials of cloze-1, cloze-01, cloze-012 tasks for {voice}\'s inventory')
    for i in range(CLOZE_TRIALS):

        def infer(candidates, answer, bppresults, mppresults):
            bppbestconf, bppbestpred = float("-inf"), None
            mppbestconf, mppbestpred = float("-inf"), None

            for candidate, guess in candidates:
                bppconf = model.bpp(candidate, e)
                mppconf = model.mpp(candidate, e)
                if bppconf > bppbestconf:
                    bppbestconf = bppconf
                    bppbestpred = guess
                if mppconf > mppbestconf:
                    mppbestconf = mppconf
                    mppbestpred = guess

            bppresults.append(int(answer == bppbestpred))
            mppresults.append(int(answer == mppbestpred))

        # ----------------------------------------------------------------------
        # cloze-1

        # copy so as not to delete from total inv
        cloze1_inv = list(total_vowel_inv)
        candidates = []
        i = random.randint(0, len(cloze1_inv) - 1)
        answer = set([vowel_inv[i][1]])
        del cloze1_inv[i]
        vowels = [v for fs,v in cloze1_inv]

        for v in inv.keys():
            if v not in vowels:
                candidates.append((cloze1_inv + [(inv[v], v)], set([v])))

        infer(candidates, answer, bpp1s, mpp1s)

        # ----------------------------------------------------------------------
        # cloze-01

        cloze01_inv = list(total_vowel_inv)
        candidates = []
        del_count = random.randint(0, 1)
        answer = set()
        for i in range(del_count):
            i = random.randint(0, len(cloze01_inv) - 1)
            answer.add(vowel_inv[i][1])
            del cloze01_inv[i]
        vowels = [v for fs,v in cloze01_inv]

        for v in inv.keys():
            if v not in vowels:
                candidates.append((cloze01_inv + [(inv[v], v)], set([v])))
        candidates.append((cloze01_inv, set([])))

        infer(candidates, answer, bpp01s, mpp01s)

        # ----------------------------------------------------------------------
        # cloze-012

        cloze012_inv = list(total_vowel_inv)
        candidates = []
        del_count = random.randint(0, 2)
        answer = set()
        for i in range(del_count):
            i = random.randint(0, len(cloze012_inv) - 1)
            answer.add(vowel_inv[i][1])
            del cloze012_inv[i]
        vowels = [v for fs,v in cloze012_inv]

        candidates.append((cloze012_inv, set([])))
        for v in inv.keys():
            if v not in vowels:
                candidates.append((cloze012_inv + [(inv[v], v)], set([v])))
        for v1, v2 in combinations(inv.keys(), 2):
            if v1 not in vowels and v2 not in vowels:
                candidates.append((cloze012_inv + [(inv[v], v)], set([v1, v2])))

        infer(candidates, answer, bpp012s, mpp012s)

    # NOTES
    # this is still pretty meaningless, partly because with the aggregated
    # inventory, there's not tough choice of phoneme (if there's one removed,
    # there's usually only one left in the inv to choose from).
    # so candidates should also have the becker corpus vowels added? or create
    # indic/becker equivalence and then add candidates between the union of the two
    # sets.
    # or lastly, include instances of indic data's inventories in the training of
    # the vowel typology model to make sure these inventories have greater
    # probability mass.

    print(f'\tBPP on cloze1 scores acc = {sum(bpp1s) / CLOZE_TRIALS * 100}%')
    print(f'\tBPP on cloze01 scores acc = {sum(bpp01s) / CLOZE_TRIALS * 100}%')
    print(f'\tBPP on cloze012 scores acc = {sum(bpp012s) / CLOZE_TRIALS * 100}%')
    print(f'\tMPP on cloze1 scores acc = {sum(mpp1s) / CLOZE_TRIALS * 100}%')
    print(f'\tMPP on cloze01 scores acc = {sum(mpp01s) / CLOZE_TRIALS * 100}%')
    print(f'\tMPP on cloze012 scores acc = {sum(mpp012s) / CLOZE_TRIALS * 100}%')

    all_results.append((voice, "cloze1", sum(bpp1s) / CLOZE_TRIALS * 100, sum(mpp1s) / CLOZE_TRIALS * 100))
    all_results.append((voice, "cloze01", sum(bpp01s) / CLOZE_TRIALS * 100, sum(mpp01s) / CLOZE_TRIALS * 100))
    all_results.append((voice, "cloze012", sum(bpp012s) / CLOZE_TRIALS * 100, sum(mpp012s) / CLOZE_TRIALS * 100))


# TODO ?
# correct the vowel label discrepancies between indic and becker, enabling
#   proper cloze tasks (with a bigger class of candidate inventories)
# classification tasks (vowel, speaker, dialect)
#   through model inference
# cloze and classification tasks neural net for comparison

print()
print(str(("VOICE", "TASK", "BPP RESULT", "MPP RESULT")))
print("\n".join(map(str, all_results)))
