"""Functions to download, preprocess data and create bandit problems"""

import urllib.request
import os
import validators
import numpy as np
import pandas as pd

def download_data(file_url,
                  download_dir = "datasets",
                  file_name = None):
    """
    This function downloads a file over HTTP
    Args:
        file_url: The url of the file to download
        download_dir: The sirectory to save the downloaded files to
        file_name: Name of the downloaded file. Uses the file name from the url by default      
    """
    if not validators.url(file_url):
        raise RuntimeError("Invalid URL {}".format(file_url))
    
    if file_name == None:
        file_name = file_url.split('/')[-1]
    
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
    
    urllib.request.urlretrieve(file_url, os.path.join(download_dir, file_name))

def one_hot_encoder(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def sample_mushroom_data(download_file=False,
                         file_name=os.path.join("datasets","mushroom.data"),
                         num_contexts=100,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """Samples bandit game from Mushroom UCI Dataset.
    Args:
    file_name: Route of file containing the original Mushroom UCI dataset.
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    r_noeat: Reward for not eating a mushroom.
    r_eat_safe: Reward for eating a non-poisonous mushroom.
    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
    Returns:
    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
    opt_vals: Vector of expected optimal (reward, action) for each context.
    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
    """

    # first two cols of df encode whether mushroom is edible or poisonous
    if download_file:
        file_path = os.path.split(file_name)
        download_data(file_url="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",download_dir=os.path.join(file_path[:-1]), file_name=file_path[-1])
    df = pd.read_csv(file_name, header=None)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

    contexts = df.iloc[ind, 2:]
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.reshape((num_contexts, 1))

    # compute optimal expected reward and optimal actions
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
        r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
    # actions: no eat = 0 ; eat = 1
        opt_actions = df.iloc[ind, 0]  # indicator of edible
    else:
    # should always eat (higher expected reward)
        opt_actions = np.ones((num_contexts, 1))

    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals