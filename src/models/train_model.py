import tensorflow as tf
from src.data.make_dataset import load_dataset
from tensorflow.keras.layers import (Bidirectional, Concatenate, Dense, Dropout, Embedding,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

def main():
    """
    Train and save the model
    """
    input_data, output_data, in_lang, out_lang = load_dataset()

    # Create sequences for the output data
    target_data = [[
        output_data[n][i + 1] for i in range(len(output_data[n]) - 1)
    ] for n in range(len(output_data))]
    target_data = tf.keras.preprocessing.sequence.pad_sequences(
        target_data, maxlen=out_lang.max_length, padding="post")
    target_data = target_data.reshape((target_data.shape[0],
                                       target_data.shape[1], 1))

    # Shuffle all of the data
    p = np.random.permutation(len(input_data))
    input_data = input_data[p]
    output_data = output_data[p]
    target_data = target_data[p]

    # Get the model
    build_model(input_data, in_lang, out_lang)

    # Train the model

    # Save the model


def train_model(input_data, output_data, in_lang, out_lang):
    """
    Trains model and returns data
    """
    pass


def save_model(model):
    """
    Save model to disk
    """
    pass


def build_model(input_data, input_lang, target_lang):
    """
    Builds and compiles the model
    """
    BUFFER_SIZE = len(input_data)
    BATCH_SIZE = 128
    embedding_dim = 1024
    units = 256
    vocab_in_size = len(input_lang.word2idx)
    vocab_out_size = len(target_lang.word2idx)

    # Check if GPU processing is enabled, select optimized layers
    try:
        from tensorflow.keras.layers import CuDNNGRU
        Rec_layer = CuDNNGRU
    except:
        from tensorflow.keras.layers import GRU
        Rec_layer = GRU

    # Create encoder layer
    encoder_inputs = Input(shape=(input_lang.max_length, ))
    encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)

    encoder_gru = Bidirectional(
        Rec_layer(units=units, return_sequences=True, return_state=True))
    encoder_out, fstate, bstate = encoder_gru(encoder_emb(encoder_inputs))
    encoder_state = Concatenate()([fstate, bstate])

    # Create decoder layer
    decoder_inputs = Input(shape=(None, ))
    decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
    decoder_gru = Rec_layer(
        units=units * 2, return_sequences=True, return_state=True)
    decoder_gru_out, _ = decoder_gru(
        decoder_emb(decoder_inputs), initial_state=encoder_state)

    # Create inference layers
    decoder_d1 = Dense(units, activation="relu")
    decoder_d2 = Dense(vocab_out_size, activation="softmax")
    decoder_out = decoder_d2(
        Dropout(rate=.2)(decoder_d1(Dropout(rate=.2)(decoder_gru_out))))

    # Compile the model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_out)
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    return model


if __name__ == '__main__':
    main()
