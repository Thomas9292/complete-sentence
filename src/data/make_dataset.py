# -*- coding: utf-8 -*-
import logging
import os
import pickle
import re
from pathlib import Path

import click
import tensorflow as tf

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.features.build_features import LanguageIndex


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    with open(os.path.join(input_filepath, 'common_phrases.txt'), 'r') as file:
        corpus = file.readlines()

    processed_data, processed_df = preprocess_dataset(corpus)

    import pdb
    pdb.set_trace()

    # Save files
    with open(os.path.join(output_filepath, 'processed_data.pkl'),
              'wb') as file:
        pickle.dump(processed_data, file)

    processed_df.to_csv(os.path.join(output_filepath, 'processed_df.csv'))



def preprocess_dataset(corpus):
    processed_data = []
    for text_message in corpus:
        preprocessed_msg = preprocess_message(text_message)

        if len(preprocessed_msg) > 0:
            for i in range(1, len(preprocessed_msg)):
                # adding a start and an end token to the sentence
                # so that the model know when to start and stop predicting.
                left_input = '<start> ' + preprocessed_msg[:i + 1] + ' <end>'
                right_output = '<start> ' + preprocessed_msg[i + 1:] + ' <end>'

                processed_data.append([left_input, right_output])

                # Skip messages with digits, as they can be phone numbers, times, etc. (Non-generalizable)
                if any(i.isdigit() for line in left_input
                       for i in line) or any(i.isdigit()
                                             for line in right_output
                                             for i in line):
                    processed_data.pop()

    processed_df = pd.DataFrame(processed_data, columns=['input', 'output'])
    return processed_data, processed_df


def preprocess_message(w):
    """
    Performs preprocessing on single message in string datatype
    """
    w = w.lower().strip()

    # Remove links and email addresses
    w = [
        word for word in w.split(' ')
        if ('@' not in word and 'html' not in word and 'http' not in word)
    ]
    w = ' '.join(w)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    return w


def load_dataset():
    """
    Transform raw data into machine readable data
    """
    processed_data, processed_df = get_saved_data()

    # Create language indexes
    in_lang = LanguageIndex(in_phr for in_phr, out_phr in processed_data)
    out_lang = LanguageIndex(out_phr for in_phr, out_phr in processed_data)

    # Create input and output vectors
    input_data = [[in_lang.word2idx[s] for s in in_phr.split(' ')]
                  for in_phr, out_phr in processed_data]
    output_data = [[out_lang.word2idx[s] for s in out_phr.split(' ')]
                   for in_phr, out_phr in processed_data]

    # Pad sequences
    input_data = tf.keras.preprocessing.sequence.pad_sequences(
        input_data, maxlen=in_lang.max_length, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(
        output_data, maxlen=out_lang.max_length, padding="post")

    return input_data, output_data, in_lang, out_lang


def get_saved_data():
    """
    Get saved data
    """
    project_dir = Path(__file__).resolve().parents[2]
    output_filepath = os.path.join(project_dir, 'data', 'processed')

    with open(os.path.join(output_filepath, 'processed_data.pkl'),
              'rb') as file:
        processed_data = pickle.load(file)

    processed_df = pd.read_csv(
        os.path.join(output_filepath, 'processed_df.csv'))

    return processed_data, processed_df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
