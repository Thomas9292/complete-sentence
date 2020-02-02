complete-sentence
==============================

A machine learning project to predict how to complete a sentence

The prediction is done using an encoder-decoder model. The recurrent layer is a bidirectional GRU (Gated Recurrent Unit) in the encoder, and a normal GRU (of twice the size) in the decoder layer.

![Network architecture](https://imgur.com/download/w5udTeM)

Loosely based on the Gmail smart compose function, and the Google AI blogs describing it (see references).

Results
------------
The following predictions were made by the model, and can be repeated by running the visualize script.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Input</th>
      <th>Predicted sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Can you let me know</td>
      <td>know if this works</td>
    </tr>
    <tr>
      <th>1</th>
      <td>thanks fo</td>
      <td>r your email .</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sorry for the late</td>
      <td>reply .</td>
    </tr>
    <tr>
      <th>3</th>
      <td>After careful consideration</td>
      <td>we have decided</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I would apprec</td>
      <td>id we can find a solution soon .</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I know that is a lot to take in</td>
      <td>, so let me know if anything i ve said doesn t make sense .</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sorry it s been so long since my</td>
      <td>last email .</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Please keep</td>
      <td>me posted .</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Just a quic</td>
      <td>k reminder that</td>
    </tr>
    <tr>
      <th>9</th>
      <td>i m sorry</td>
      <td>hope we had a chance to chat at the convention .</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I m afraid we</td>
      <td>need to cancel our meeting .</td>
    </tr>
    <tr>
      <th>11</th>
      <td>What are y</td>
      <td>me posted .</td>
    </tr>
    <tr>
      <th>12</th>
      <td>What exac</td>
      <td>ly do you think ?</td>
    </tr>
    <tr>
      <th>13</th>
      <td>I hope you</td>
      <td>had a great trip .</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Could you please expl</td>
      <td>ain what you are available</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sorry I couldn t be</td>
      <td>of more help .</td>
    </tr>
  </tbody>
</table>

Usage
------------
The file structure can be found below. To run some predictions:

Install the requirements:
```
$ pip install -r requirements.txt
```

Set custom sentences in `src\visualization\visualize.py` and run the visualization
```
$ python src/visualization/visualize.py
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
