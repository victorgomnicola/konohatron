from bpe import Encoder
import pickle

def get_train_sample(text, encoder):
	if len(encoder.tokenize(text)) <= 100:
		return small_text_sample(text, encoder)


def small_text_sample(text, encoder):
	X = []
	Y = []

	tokens = next(encoder.transform([text]))
	for i in range(1, len(tokens)-1):
		x = tokens[0:i] + [1 for _ in range(100-i)]
		X.append(x)
		Y.append(tokens[i])

	return X, Y

#open the encoder
encoder_file = './dataset/byte_pair_encoder.pkl'
encoder = pickle.load(open(encoder_file, 'rb'))

#open the dataset
sample_file = './dataset/training.pkl'
#dataset = pickle.load(open(sample_file, 'rb'))

test = 'testando a amostra para ver o que acontece com essa coisa'

X, Y = get_train_sample(test, encoder)
print('tamanho:', len(X[0]))
print(next(encoder.inverse_transform(X)))
print(Y)


