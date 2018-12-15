
import os.path
import numpy as np
import pickle

class IndicFormantData:

    # fields:
    #   data   - loaded formant data
    #   N      - total sample count
    #   aggr   - tuple of (aggregated formant samples, aggregated sample labels)
    #   lookup - tuple of (dict(vowel, int), np.array(vowels))
    #              (allows vowel to int and int to vowel conversion)

    # the data dict has in each value a label/samples tuple
    # the label array is a 1d array of vowel labels (strings)
    # the samples matrix is as folllows:
    #
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
    # |    |             |  |
    # ...
    #
    # voice_samples[sample, timestep, formant]
    # sample   - from 1 to 37901
    # timestep - from 1 to 10
    # formant  - from 1 to 3

    # --------------------------------------------------------------------------
    # consts

    VOWEL_INVENTORY_FILE = "data/vowel_inv.log"
    VOWEL_FORMANT_FILE = "data/vowel_formants.log"
    INDIC_DATA_STORE = "data/dump/indic_data.pickle"

    # ownas ow aInas onas inas uy enas Anas unas ay (totalling 2.2%)
    UNCOMMON = set(['ownas', 'ow', 'aInas', 'onas', 'inas', 'uy', 'enas', \
        'Anas', 'unas', 'ay'])

    SAMPA_LOOKUP = { "aI": "aI", "aU": "aʊ", "i": "i", "e": "e", "o": "o", \
        "ow": "oʷ", "u": "u", "A": "ɑ", "E": "ɛ", "9r": "œ̝" }

    # --------------------------------------------------------------------------
    # data loading

    def __init__(self):
        # this file takes a bit to load so let's cache the loaded data model
        if os.path.isfile(self.INDIC_DATA_STORE):
            with open(self.INDIC_DATA_STORE, 'rb') as indic_data_file:
                obj = pickle.load(indic_data_file)
                self.data = obj.data
                self.N = obj.N
                self.aggr = obj.aggr
                self.lookup = obj.lookup
        else:
            #       voice                     labels                  samples
            # dict("lang_dialect_dataset" -> (np.array(vowel labels), np.array(formant samples)))
            data = {}
            N = 0
            aggr_data = np.array([])
            aggr_labs = np.array([])
            v2i, i2v = {}, []

            with open(self.VOWEL_INVENTORY_FILE) as vowels:
                for i, v in enumerate(vowels.readlines()):
                    if ':' not in v and '=' not in v:
                        v2i[v] = i
                        i2v.append(v)

            with open(self.VOWEL_FORMANT_FILE) as formants:
                for row in formants.readlines():
                    label, *entries = row.split()
                    if label == "label":
                        continue

                    # voice = "lang_dialect_dataset", descriptor = "utterance-vowel-instance"
                    voice, descriptor = label.rsplit('_', 1)
                    uttr_number, sample_label, instance = descriptor.split('-')
                    # ignore diacritics that don't affect formants
                    #   : for lengthened, = for syllabic
                    sample_label = sample_label.replace(":", "").replace("=", "")
                    # _r is a raised vowel, whose contrastivity we maintain
                    # same for _w as labialisation
                    if sample_label in UNCOMMON:
                        # strip least common 2% of vowels from data
                        continue
                    # convert to a common IPA symbol
                    sample_label = SAMPA_LOOKUP[sample_label]

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

            self.data = data
            self.N = N
            self.aggr = (aggr_data, aggr_labs)
            self.lookup = (v2i, i2v)

            with open(self.INDIC_DATA_STORE, 'wb') as indic_data_file:
                pickle.dump(self, indic_data_file)

        print()
        print("Formant data loaded!")
        print(str(len(self.data.keys())) + " unique voice/language/dialect tuples")
        print(str(self.N) + " individual vowel data points")

        samples, intervals, formants = self.aggr[0].shape
        flattened = self.aggr[0].reshape(samples * intervals, formants)
        avgs = np.mean(flattened, axis=0)
        stds = np.std(flattened, axis=0)
        print("f1 ~ " + str(int(avgs[0])) + " +/- " + str(int(stds[0])))
        print("f2 ~ " + str(int(avgs[1])) + " +/- " + str(int(stds[1])))
        print("f3 ~ " + str(int(avgs[2])) + " +/- " + str(int(stds[2])))

        ## vowel frequency analysis
        #from ascii_graph import Pyasciigraph
        #print()
        #freqs = {}
        #for v in self.aggr[1]:
        #    if v not in freqs:
        #        freqs[v] = 0
        #    freqs[v] += 1
        #def format_label(v, p):
        #    a = str(v)
        #    b = "{0:.2f}".format(p * 100) + "%"
        #    c = 15 - len(a) - len(b) # buffer spaces for alignment
        #    return a + (" " * c) + b
        #freq_graph = [(format_label(v, float(c) / self.N), int(c)) for v, c in freqs.items()]
        #graph = Pyasciigraph()
        #for line in graph.graph('indic vowel frequency', freq_graph):
        #    print(line)
        #print("vowels making up the bottom 2% of the data:")
        #running_total = 0.0
        #for c, v in sorted([(c, v) for v, c in freqs.items()]):
        #    running_total += float(c) / self.N
        #    print(v, end=' ')
        #    if running_total >= 0.02:
        #        break
        #print(f'(totalling {running_total*100}%)')


if __name__ == '__main__':
    formant_data = IndicFormantData()
