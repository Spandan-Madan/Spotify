Spotify
=================
Repo for the spotify project for capstone course (AC297R, 2018 Spring).

Team Members:

* Spandan Madan
* Mehul Smriti Raje
* Timothy Lee
* Benjamin Sanchez-Lengeling

**Group TF:** Patrick Ohiomoba

Directory Structure:



Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── AC297R             <- AC297r related information (submission, meeting logs, etc)
    │   ├── meeting_notes  <- Keep track of AC297r related meeting logs
    │   ├── ignite         <- AC297r Ignite presentations
    │   └── submissions    <- Milestone and final submissions
    │  
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



# Running Pooling model
```
# Make sure that modules/ is in your path. use sys.append for it if needed.
from pooling_model import pooled_songs
playlist = [('The Scientist','Coldplay'),('Immigrant Song','Led Zeppelin'),('T.N.T.','AC/DC')]
VERBOSE = False #Don't print out things
pickle_files_folder = <path of the folder where you've moved word2vec models and other pickle files
songs_pool, most_simiilar_artists = pooled_songs(playlist,VERBOSE,pickle_files_folder)
```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
