from bpe import Encoder
import pickle


def create_byte_pair_encoding(corpus, vocab_size = 100, pb = 0.88):
	encoder = Encoder(vocab_size, pb)  # params chosen for demonstration purposes
	encoder.fit(corpus)

	return encoder

sample_file = './dataset/training_2.pkl'
corpus = pickle.load(open(sample_file, 'rb'))


encoder = create_byte_pair_encoding(corpus, vocab_size=1000)


#persists the encoder
encoder_file = './dataset/byte_pair_encoder.pkl'
pickle.dump(encoder, open(encoder_file, 'wb'))

for c in corpus:
	print(c)

#test the persistency
encoder = pickle.load(open(encoder_file, 'rb'))


example = "abacaxi"
print(encoder.tokenize(example))

print(next(encoder.transform([example])))

print(next(encoder.inverse_transform(encoder.transform([example]))))
