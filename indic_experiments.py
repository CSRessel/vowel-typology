
import os.path
import pickle
import numpy as np
from cotterell_model import CotterellModel

VOWEL_FORMANT_FILE = "data/vowel_formants.log"
INDIC_DATA_STORE = "data/indic_data.pickle"

model = CotterellModel()

# data importing

#  <- formant ->
#  _____________
# |\             \  ^
# | \             \  \
# |  \             \  \ time
# |   \             \  \
# |    \_____________\  v
# |    |             |  ^
# |    |             |  |
# |    |             |  | sample
# |    |             |  |
# |    |             |
# ...
#
# voice_samples[sample, timestep, formant]
# sample   - from 1 to 37901
# timestep - from 1 to 10
# formant  - from 1 to 3

# this file takes a bit to load so let's cache the loaded data model
if os.path.isfile(INDIC_DATA_STORE):
    with open(INDIC_DATA_STORE, 'rb') as indic_data_file:
        data, N, aggr_data, aggr_labs = pickle.load(indic_data_file)
else:
    #       voice                     labels                  samples
    # dict("lang_dialect_dataset" -> (np.array(vowel labels), np.array(formant samples)))
    data = {}
    N = 0
    aggr_data = np.array([])
    aggr_labs = np.array([])

    with open(VOWEL_FORMANT_FILE) as formants:
        for row in formants.readlines():
            label, *entries = row.split()
            if label == "label":
                continue

            # voice = "lang_dialect_dataset", descriptor = "utterance-vowel-instance"
            voice, descriptor = label.rsplit('_', 1)
            uttr_number, sample_label, instance = descriptor.split('-')

            sample_formants = np.zeros((10, 3))
            for i, entry in enumerate(entries):
                timestep = int(i/3)
                formant = i%3
                sample_formants[timestep, formant] = float(entry)

            if voice in data:
                sample_labels, samples = data[voice]
                samples = np.append(samples, np.expand_dims(sample_formants, axis=0), axis=0)
                sample_labels = np.append(sample_labels, [sample_label])
            else:
                samples = np.expand_dims(sample_formants, axis=0)
                sample_labels = np.array([sample_label])
            data[voice] = (sample_labels, samples)
            N += 1

            if aggr_data.size > 0:
                aggr_data = np.append(aggr_data, np.expand_dims(sample_formants, axis=0), axis=0)
                aggr_labs = np.append(aggr_labs, [sample_label])
            else:
                aggr_data = np.expand_dims(sample_formants, axis=0)
                aggr_labs = np.array([sample_label])

    with open(INDIC_DATA_STORE, 'wb') as indic_data_file:
        pickle.dump((data, N, aggr_data, aggr_labs), indic_data_file)

print()
print("Formant data loaded!")
print(str(len(data.keys())) + " unique voice/language/dialect tuples")
print(str(N) + " individual vowel data points")

# evaluation tasks
# ----
# cross entropy for held out vowel instances
#   simple-exp model run on sets of these vowels?
#     so just try creating zsc and pca data sets from each voice's phonemes
#     then print model probability of each voice's vowel inventory
#     then print model xent of each speaker's vowel inv's when multiple languages available
#
# cloze tasks
#   through model inference
#   through neural net
# classification tasks (vowel, speaker, dialect)
#   through model inference
#   through neural net
