from bpe import Encoder
import pickle
import random
from tqdm import tqdm

sample_file = 'training.pkl'
corpus = pickle.load(open(sample_file, 'rb'))


encoder = Encoder(100, pct_bpe=0.88)  # params chosen for demonstration purposes
encoder.fit(corpus)

example = ""
print(encoder.tokenize(example))

print(next(encoder.transform([example])))

print(next(encoder.inverse_transform([[7]])))

