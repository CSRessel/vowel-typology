
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import KFold
from vowel_typology_model import VowelTypologyModel
from indic_formant_data import IndicFormantData

model = VowelTypologyModel()
formant_data = IndicFormantData()

aggr_data, aggr_labs = formant_data.aggr
unknown = set()
# remove samples with labels unknown w.r.t. the becker vowel corpus annotations
# (backwards so we don't offset indices)
for i, (sample, label) in enumerate(zip(reversed(aggr_data), reversed(aggr_labs))):
    i = formant_data.N - 1 - i
    if label in unknown:
        aggr_data = np.delete(aggr_data, i, axis=0)
        aggr_labs = np.delete(aggr_labs, i)

# TODO: hardcoded phoneme label conversions?
#fest2ipa = dict()

# delete the third value (f3) along axis=2
aggr_data = np.delete(aggr_data, 2, axis=2)
zsc_data = zscore(aggr_data, axis=1)
kf = KFold(n_splits=10)

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

def e(v):
    return v

# 10 fold evaluation of probability of the fold's validation set's assigned
# probability mass
bestbppresult = float("inf")
bestmppresult = float("inf")
print()
for i, (train_idx, val_idx) in enumerate(kf.split(zsc_data, aggr_labs)):
    print("fold " + str(i+1) + "\tstarting at " + str(val_idx[0]) + "\twith probability = ", end='')
    val_x = zsc_data[val_idx]
    val_y = aggr_labs[val_idx]
    vowel_inv = preprocess(val_x, val_y)

    def e(v):
        return v
    bppresult = model.bpp(inv=vowel_inv, embeddings=e)
    mppresult = model.mpp(inv=vowel_inv, embeddings=e)
    print(str(int(bppresult)) + " (bpp) / ", end='')
    print(str(int(mppresult)) + " (mpp)")
    if bppresult < bestbppresult:
        bestbppresult = bppresult
    if mppresult < bestmppresult:
        bestmppresult = mppresult

# essentially zero probability, because
#   A) the model doesn't do focalization correctly
#   B) these vowel inventories are real big, even in the KFold
# need to fix the VowelTypologyModel, and consider removing the least common
# vowels from this dataset?
print("\tBPP assigns probability = " + str(bestbppresult))
print("\tMPP assigns probability = " + str(bestmppresult))

# cloze tasks
#   through model inference
#   through neural net

# classification tasks (vowel, speaker, dialect)
#   through model inference
#   through neural net
