import os
from pathlib import Path

import numpy as np

import pandas as pd
from src.data.make_dataset import load_dataset
from src.models.utils import load_model


def main():
    """
    Load model and start prediction
    """
    # Load the language indices and models
    _, _, in_lang, out_lang = load_dataset()

    encoder_model = load_model('encoder_model')
    inf_model = load_model('inf_model')

    # Initialize the prediction engine
    engine = PredictionEngine(encoder_model, inf_model, in_lang, out_lang)

    # Predict
    test = [
        'Can you let me know',
        'thanks fo',
        'Sorry for the late',
        'After careful consideration',
        'I would apprec',
        "I know that is a lot to take in",
        "Sorry it s been so long since my ",
        "Please keep",
        "Just a quic",
        "i m sorry",
        "I m afraid we ",
        "What are y",
        "What exac",
        "I hope you ",
        "Could you please expl",
        "Sorry I couldn t be",
    ]

    # Make predictions
    output = []
    for input in test:
        output.append({
            "Input ": input,
            "Predicted sentence ": engine.predict(input)
        })

    # Print the answers
    pd.set_option('display.max_colwidth', -1)
    answers = pd.DataFrame.from_dict(output)
    print(answers.head(len(test)))

    # Get html of table
    project_dir = Path(__file__).resolve().parents[2]
    print(answers.to_html(
        os.path.join(project_dir, 'reports', 'predictions.html')))


class PredictionEngine:
    def __init__(self, encoder_model, inf_model, in_lang, out_lang):
        self.encoder_model = encoder_model
        self.inf_model = inf_model
        self.in_lang = in_lang
        self.out_lang = out_lang

    def predict(self, input_sentence):
        """
        Predict the ending of input sentence
        """
        # Convert input sentence to vector
        sentence_vector = self.sentence2vector(input_sentence.lower())

        # Send input sentence through encoder model
        inf_model = self.inf_model
        [enc_out, state] = self.encoder_model.predict(x=sentence_vector)

        cur_vec = np.zeros((1, 1))
        cur_vec[0, 0] = self.out_lang.word2idx['<start>']
        cur_word = "<start>"
        output_sentence = ""

        # Predict words while not <end> or max_length of prediction
        i = 0
        while cur_word != "<end>" and i < self.out_lang.max_length - 1:
            i += 1
            if cur_word != "<start>":
                output_sentence = output_sentence + " " + cur_word
            input = [cur_vec, state]
            [out_vec, state] = inf_model.predict(x=input)
            cur_vec[0, 0] = np.argmax(out_vec[0, 0])
            cur_word = self.out_lang.idx2word[cur_vec[0, 0]]

        return output_sentence

    def sentence2vector(self, input_sentence):
        """
        Translates the input sentence to vector using in_lang
        """
        # Initialize vector
        vector = np.zeros(self.in_lang.max_length)

        # Translate words to ids
        sentence_ids = [
            self.in_lang.word2idx[word] for word in input_sentence.split()
        ]

        # Input in vector and shape
        for i, word_id in enumerate(sentence_ids):
            vector[i] = word_id
        vector = vector.reshape([1, len(vector)])

        return vector


if __name__ == '__main__':
    main()
