"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""
import keras
import tensorflow as tf
from tensorflow import keras
from keras_position_wise_feed_forward import FeedForward
from tensorflow.keras import layers
from bpemb import BPEmb
import wikipedia
import numpy as np
import threading
import gc
"""
## Implement multi head self attention as a Keras layer
"""


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Dense(embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.max_len = maxlen

    def call(self, x):
        #maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



vocab_size = 1000
bpe_dim = 25
encoder = BPEmb(lang= 'pt', vs = vocab_size, dim = bpe_dim)
text_samples = []

def sample(number, n=4, max_len = 300):

    global text_samples

    s = []
    while(len(s)<n):
        print(str(number)+"-Generating samples: "+str(round(len(s)/n, 2)), end = "\r")
        try:
            text = wikipedia.summary(wikipedia.random(1))
            text_ids = encoder.encode_ids(text)
            #print(text)
            #print("--------------", len(text_ids))
            if(len(text_ids)< max_len) and (len(text_ids)!=0):
                s.append(text_ids)
        except:
            pass
    
    text_samples = text_samples + s 

    #return s

def sample_multthread(n = 32, n_threads = 8):

    global text_samples
    text_samples = []
    threads = []
    assert(n % n_threads == 0)
    
    for i in range(n_threads):
        t = threading.Thread(target=sample, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert(len(text_samples) == n)

    return text_samples.copy()

def preprocess(text_list, max_len = 300):

    X = []
    Y = []

    for text in text_list:

        for i in range(1, len(text)-1):

            X.append(np.concatenate((encoder.vectors[text[0:i]], np.zeros(shape= (max_len-i, bpe_dim), dtype = np.float32)), axis = 0))
            Y.append(text[i])
            #print(X[-1].shape)

    return np.array(X), np.array(Y)


max_len = 300
embed_dim = 256  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 1024  # Hidden layer size in feed forward network inside transformer


inputs = layers.Input(shape=(max_len, bpe_dim))
embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
x = embedding_layer(inputs)
#transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
for _ in range(5):
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
#model.load_weights('transformer.h5')

cosine_schedule = tf.keras.experimental.CosineDecayRestarts(1e-5,10000)
print('cosseno')
opt = keras.optimizers.Adam(learning_rate=cosine_schedule)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])


for epoch in range(10000):
    X,Y = preprocess(sample_multthread(), max_len = max_len)
    model.fit(X, Y, batch_size = 128)
    model.save_weights('transformer.h5')
    gc.collect()


