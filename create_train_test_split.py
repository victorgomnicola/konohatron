from bpe import Encoder
import pickle

def get_train_sample(text, encoder, window_size = 200):
	if len(encoder.tokenize(text)) <= window_size:
		return small_text_sample(text, encoder, window_size)
	else:
		return [], []


def small_text_sample(text, encoder, window_size = 200):
	X = []
	Y = []

	
	tokens = next(encoder.transform([text]))
	for i in range(1, len(tokens)-1):
		x = tokens[0:i] + [1 for _ in range(window_size-i)]
		X.append(x)
		Y.append(tokens[i])

	return X, Y

#open the encoder
encoder_file = './dataset/byte_pair_encoder.pkl'
encoder = pickle.load(open(encoder_file, 'rb'))

#open the dataset
#sample_file = './dataset/training.pkl'
sample_file = './dataset/training_2.pkl'
corupus = pickle.load(open(sample_file, 'rb'))

#test = 'testando a amostra para ver o que acontece com essa coisa'


X_train, Y_train = [], []

for c in corupus:
	#print(c)
	X, Y = get_train_sample(c, encoder, window_size=500)
	#print('tamanho:', len(X[0]))
	X_train += X
	Y_train += Y

'''
for t, o in zip(encoder.inverse_transform(X_train), X_train):
	print('original:', o)
	print('traduzido:', t)
'''

print('number of instances:', len(X_train))

training_file = './dataset/training_encoded_X.pkl'
pickle.dump(X_train, open(training_file, 'wb'))

training_file = './dataset/training_encoded_Y.pkl'
pickle.dump(Y_train, open(training_file, 'wb'))