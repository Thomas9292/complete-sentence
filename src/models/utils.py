import os
from pathlib import Path

import tensorflow as tf


def save_model(name, model):
    """
    Save model to disk under specified name
    """
    # Load path
    project_dir = Path(__file__).resolve().parents[2]
    model_path = os.path.join(project_dir, 'models', name + '.h5')

    # Save model
    model.save(model_path)


def load_model(name):
    """
    Load model from disk at specified path
    """
    # Load path
    project_dir = Path(__file__).resolve().parents[2]
    model_path = os.path.join(project_dir, 'models', name + '.h5')

    # Load model
    model = tf.keras.models.load_model(model_path)
    return model
