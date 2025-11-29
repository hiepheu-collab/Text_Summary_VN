import os
import pickle
import tensorflow as tf
import numpy as np
from keras.saving import register_keras_serializable
from keras.layers import Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize
import re
import unicodedata

@register_keras_serializable()
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(inputs) + positions

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs, mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model_cast = tf.cast(d_model, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model_cast) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }

MODEL_PATH = "summary_transformer_optimized1.keras"
X_TOKENIZER_PATH = "x_tokenizer_tranformer.pkl"
Y_TOKENIZER_PATH = "y_tokenizer_tranformer.pkl"
MAX_TEXT_LEN = 600
MAX_SUMMARY_LEN = 50

def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)
x_tokenizer = load_tokenizer(X_TOKENIZER_PATH)
y_tokenizer = load_tokenizer(Y_TOKENIZER_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
        "CustomSchedule": CustomSchedule
    }
)

def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = "".join(ch if (ch.isalpha() or ch.isspace() or ch.isdigit() or ch in ['/', '-']) else " " for ch in text)
    text = text.lower()
    tokens = word_tokenize(text, format="text").split()
    text = ' '.join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def beam_search_decoder(input_seq, k=3, max_len=MAX_SUMMARY_LEN):
    start_id = y_tokenizer.word_index.get('startseq')
    end_id = y_tokenizer.word_index.get('endseq')
    sequences = [[0.0, [start_id]]]
    for _ in range(max_len - 1):
        all_candidates = []
        for score, seq in sequences:
            if seq[-1] == end_id:
                all_candidates.append((score, seq))
                continue
            dec_input = pad_sequences([seq], maxlen=max_len-1, padding='post')
            preds = model.predict([input_seq, dec_input], verbose=0)
            pred_probs = preds[0, len(seq)-1, :]
            top_k_idx = np.argsort(pred_probs)[-k:]
            for idx in top_k_idx:
                candidate_score = score + np.log(pred_probs[idx] + 1e-10)
                all_candidates.append((candidate_score, seq + [idx]))
        sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:k]
        if all([s[1][-1] == end_id for s in sequences]): break
    best_seq = sequences[0][1]
    summary = y_tokenizer.sequences_to_texts([best_seq])[0]
    return summary.replace('startseq', '').replace('endseq', '').strip()

def abstractive_summary(text):
    clean = clean_text(text)
    seq = x_tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding='post')
    return beam_search_decoder(padded, k=3, max_len=MAX_SUMMARY_LEN)