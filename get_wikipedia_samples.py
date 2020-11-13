from bpe import Encoder
import wikipedia
import pickle
import random
from tqdm import tqdm

wikipedia.set_lang("pt")

def get_wikipedia_samples(n_samples = 500):
	#sample texts randomly
	samples = wikipedia.random(n_samples)
	text_samples = []
	#generate the sample vector and show the progress
	for s in tqdm(samples):
	#solve the disambiguation problem
		try:
			p = wikipedia.summary(s)
		except Exception as e:
			continue

		text_samples.append(p)


	return text_samples

#generate the sampled dataset
samples = []
for i in range(21):
	samples += get_wikipedia_samples()

#persists the samples
sample_file = './dataset/big_training.pkl'
pickle.dump(samples, open(sample_file, 'wb'))

#test the persistency
samples = pickle.load(open(sample_file, 'rb'))

for s in samples:
	print(s)