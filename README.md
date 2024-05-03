# pt-rec-simplex

This repository contains implementation of collaborative filtering algorithm explained in the SimpleX [paper](https://arxiv.org/abs/2109.12613), which introduced novel loss function called Cosine Contrastive Loss(CCL) as a training objective. Implemented experiment can be easily reproduced by following along the instructions in this document, as it contains functions from downloading [MovieLens dataset](https://grouplens.org/datasets/movielens/) to evaluating trained model using Streamlit demo.

1. [Project Details](#project-details)
2. [Installation](#installation)
3. [Execution](#execution)

![demo thumbnail](/demo/images/thumbnail.png)

## Project Details

```text
.
|-- main.py
|-- commands.py
|-- jobs
|   |-- __init__.py
|   |-- preprocess.py
|   |-- train.py
|   |-- evaluate.py
|   `-- config.py
|-- simplex
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   `-- trainer.py
|-- tests
|   |-- __init__.py
|   |-- test_loss.py
|   `-- test_metrics.py
`-- demo
    |-- __init__.py
    |-- pages
    `-- home.py
```

* `main.py` executes [Typer](https://typer.tiangolo.com/) app which is defined in `commands.py` module where commands that this app can take are defined. 
* `jobs` package contains implementation of jobs that the app can execute.
  * `preprocess.py` : Contains 2 jobs to preprocess MovieLens 1M and 20M dataset respectively. Both jobs are implemented since it is common practice to explore various hyperparameter combinations in smaller subset(i.e. 1M) for faster iteration, and apply the empirically best set on a main dataset(i.e. 20M).
  * `train.py` : Contains a job and corresponding methods to train the model. This job is intended to be used for hyperparameter set exploration using MovieLens 1M dataset. Detailed insights driven from group of probing experiments can be found in the demo page.
  * `evaluate.py` : Contains a job to train the model on main dataset and save files to present and share recommendation result through the demo. This job is intended to be executed after selecting fewer hyperparameter combinations from train job.
  * `config.py` : Contains configurations for the experiment(ex. file name, hyperparameter setting). Also, it contains definition of `ExecutableJobs` class used to define list of jobs that the app can execute; only the `execute` method of subclass of this class can be registered.
* `simplex` package contains any model related implementation.
  * `data.py` : Contains PyTorch Dataset class definition to make raw data to be consumed by PyTorch DataLoader and feed into the model. Also, it contains a few dataset related functionalities including random data initialization which is required at the beginning of every epoch. 
  * `model.py` : Contains implementation of SimpleX model and Cosine Contrastive Loss function.
  * `trainer.py` : Contains trainer class implementation which contains methods required for model training.
* `tests` package contains unit tests on loss and metric functions implemented in `simplex` package. As it is implemented to follow [pytest](https://docs.pytest.org/en/latest/) convention, this can be tested by single `pytest` command after installation. There is no failing test at the time of documentation. 
* `demo` contains Streamlit demo page implementation on evaluation result and trained model. Indeed, for research purpose, this feature is somewhat redundant because evaluation metric value matters the most. In production, however, this is essential feature because it demonstrates the usage and value of developed system and therefore persuades an organization to adopt the new feature.

## Installation

To get started, create virtual environment and install required libraries.

```shell
python --version  # Code tested on Python 3.10.13
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, browse list of executable jobs.

```shell
python main.py list-jobs
```

Following output is expected to be printed on console.

```text
preprocess-1m 
    Download MovieLens 1M data from source, preprocess raw data and save preprocessed data
    
preprocess-20m 
    Download MovieLens 20M data from source, preprocess raw data and save preprocessed data
    
train 
    Train a SimpleX model on a ML-1M dataset and explore various hyperparameter combinations
    
evaluate 
    Train a SimpleX model using ML-20M data with the best hyperparameter combination and evaluate model on test data
```

To check out the results only, jump directly to instruction in [web demo execution](#web-demo-execution) section after installation. Every required file to run the demo is already contained in the `demo` package.

## Execution

### run jobs

Ideal order of execution is `preprocess-1m` ,`train`, `preprocess-20m` and `evaluate`. This is because train job is iterated several times on a smaller subset of original data to determine optimal set of hyperparameter values before training model with original dataset. Jobs are run in this order by default when submitting following command without any option.

```shell
python main.py run-jobs
```

However, each job can be run separately by passing `--jobs` parameter. Value of this parameter has to be either single job name or comma-separated job names in order. 

```shell
python main.py run-jobs --jobs train
python main.py run-jobs --jobs preprocess-1m,train
```

### tensorboard monitoring

While training, log message on training loss and validation metrics are printed on the console at the end of every epoch. However, this is also recorded on the Tensorboard, whose log directory is set to be `runs/train` or `runs/evaluate` depending on the executed job. Therefore, tensorboard application executed with one of these commands will show trajectory of recorded values.

```shell
tensorboard --logdir runs/train
tensorboard --logdir runs/evaluate
```

### web demo execution

After running evaluation job, every object to be displayed on the demo would have been saved on the configured folder. However, for the sake of convenience, demo is implemented to use files that came with this repository. Following command will launch the demo to see validation metric on test data and try out recommendation result on randomly selected test users.

```shell
streamlit run demo/home.py
```
