import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

from tensorflow.keras.layers import (
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout
)

# =====================================================
# CUSTOM LAYERS
# =====================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.token_embedding = Embedding(
            vocab_size,
            embed_dim,
            mask_zero=True
        )

        self.position_embedding = Embedding(
            sequence_length,
            embed_dim
        )

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        return (
            self.token_embedding(inputs)
            + self.position_embedding(positions)
        )


@tf.keras.utils.register_keras_serializable()
class BERTBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(
                ff_dim,
                activation="gelu"
            ),
            Dense(embed_dim)
        ])

        self.norm1 = LayerNormalization()

        self.norm2 = LayerNormalization()

        self.dropout1 = Dropout(
            dropout_rate
        )

        self.dropout2 = Dropout(
            dropout_rate
        )

    def call(
        self,
        inputs,
        training=False
    ):

        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = self.norm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(x)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.norm2(
            x + ffn_output
        )

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="BERT Mask Predictor",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# LOAD FILES
# =====================================================

model = tf.keras.models.load_model(
    "models/bert_model.keras"
)

vocab = joblib.load(
    "models/vocabulary.pkl"
)

word_to_index = {
    word:i
    for i,word in enumerate(vocab)
}

# =====================================================
# TITLE
# =====================================================

st.title(
    "🧠 BERT Masked Word Predictor"
)

st.write(
    "Predict missing words using a BERT Encoder built from scratch."
)

st.markdown("---")

# =====================================================
# INPUT
# =====================================================

sentence = st.text_input(
    "Enter sentence containing 'mask'",
    value="cls i love mask learning sep"
)

# =====================================================
# PREDICT
# =====================================================

if st.button("Predict Missing Word"):

    tokens = sentence.lower().split()

    token_ids = [
        word_to_index.get(
            token,
            1
        )
        for token in tokens
    ]

    while len(token_ids) < 12:
        token_ids.append(0)

    token_ids = np.array(
        [token_ids]
    )

    mask_id = word_to_index["mask"]

    mask_position = np.where(
        token_ids[0] == mask_id
    )[0]

    if len(mask_position) == 0:

        st.error(
            "Sentence must contain 'mask'"
        )

    else:

        mask_position = mask_position[0]

        preds = model.predict(
            token_ids,
            verbose=0
        )

        predicted_id = np.argmax(
            preds[
                0,
                mask_position
            ]
        )

        predicted_word = vocab[
            predicted_id
        ]

        st.success(
            f"Predicted Word: {predicted_word}"
        )