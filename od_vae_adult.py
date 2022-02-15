import alibi
import matplotlib
# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Dense, InputLayer

from alibi_detect.od import OutlierVAE
from alibi_detect.utils.perturbation import inject_outlier_tabular
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.utils.visualize import plot_instance_score

def set_seed(s=0):
    np.random.seed(s)
    tf.random.set_seed(s)

adult = alibi.datasets.fetch_adult()
X, y = adult.data, adult.target
feature_names = adult.feature_names
category_map_tmp = adult.category_map

set_seed(0)
Xy_perm = np.random.permutation(np.c_[X, y])
X, y = Xy_perm[:,:-1], Xy_perm[:,-1]

keep_cols = [2, 3, 5, 0, 8, 9, 10]
feature_names = feature_names[2:4] + feature_names[5:6] + feature_names[0:1] + feature_names[8:11]
print(feature_names)

X = X[:, keep_cols]
print(X.shape)

category_map = {}
i = 0
for k, v in category_map_tmp.items():
    if k in keep_cols:
        category_map[i] = v
        i += 1

minmax = False

X_num = X[:, -4:].astype(np.float32, copy=False)
if minmax:
    xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
    rng = (-1., 1.)
    X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]
else:  # normalize
    mu, sigma = X_num.mean(axis=0), X_num.std(axis=0)
    X_num_scaled = (X_num - mu) / sigma

X_cat = X[:, :-4].copy()
ohe = OneHotEncoder(categories='auto')
ohe.fit(X_cat)

X = np.c_[X_cat, X_num_scaled].astype(np.float32, copy=False)

n_train = 25000
n_valid = 5000
X_train, y_train = X[:n_train,:], y[:n_train]
X_valid, y_valid = X[n_train:n_train+n_valid,:], y[n_train:n_train+n_valid]
X_test, y_test = X[n_train+n_valid:,:], y[n_train+n_valid:]
print(X_train.shape, y_train.shape, 
      X_valid.shape, y_valid.shape,
      X_test.shape, y_test.shape)

cat_cols = list(category_map.keys())
num_cols = [col for col in range(X.shape[1]) if col not in cat_cols]
print(cat_cols, num_cols)

perc_outlier = 10
data = inject_outlier_tabular(X_valid, num_cols, perc_outlier, n_std=8., min_std=6.)
X_threshold, y_threshold = data.data, data.target
X_threshold_, y_threshold_ = X_threshold.copy(), y_threshold.copy()  # store for comparison later
outlier_perc = 100 * y_threshold.sum() / len(y_threshold)
print('{:.2f}% outliers'.format(outlier_perc))

outlier_idx = np.where(y_threshold != 0)[0]
vdiff = X_threshold[outlier_idx[0]] - X_valid[outlier_idx[0]]
fdiff = np.where(vdiff != 0)[0]
print('{} changed by {:.2f}.'.format(feature_names[fdiff[0]], vdiff[fdiff[0]]))

data = inject_outlier_tabular(X_test, num_cols, perc_outlier, n_std=8., min_std=6.)
X_outlier, y_outlier = data.data, data.target
print('{:.2f}% outliers'.format(100 * y_outlier.sum() / len(y_outlier)))

X_train_ohe = ohe.transform(X_train[:, :-4].copy())
X_threshold_ohe = ohe.transform(X_threshold[:, :-4].copy())
X_outlier_ohe = ohe.transform(X_outlier[:, :-4].copy())
print(X_train_ohe.shape, X_threshold_ohe.shape, X_outlier_ohe.shape)

X_train = np.c_[X_train_ohe.todense(), X_train[:, -4:]].astype(np.float32, copy=False)
X_threshold = np.c_[X_threshold_ohe.todense(), X_threshold[:, -4:]].astype(np.float32, copy=False)
X_outlier = np.c_[X_outlier_ohe.todense(), X_outlier[:, -4:]].astype(np.float32, copy=False)
print(X_train.shape, X_threshold.shape, X_outlier.shape)

load_outlier_detector = False

filepath = './models/od_vae_adult/'  # change to directory where model is downloaded
if load_outlier_detector:  # load pretrained outlier detector
    od = load_detector(filepath)
else:  # define model, initialize, train and save outlier detector
    n_features = X_train.shape[1]
    latent_dim = 2

    encoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=(n_features,)),
          Dense(25, activation=tf.nn.relu),
          Dense(10, activation=tf.nn.relu),
          Dense(5, activation=tf.nn.relu)
      ])

    decoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=(latent_dim,)),
          Dense(5, activation=tf.nn.relu),
          Dense(10, activation=tf.nn.relu),
          Dense(25, activation=tf.nn.relu),
          Dense(n_features, activation=None)
      ])
    
    # initialize outlier detector
    od = OutlierVAE(threshold=None,  # threshold for outlier score
                    score_type='mse',  # use MSE of reconstruction error for outlier detection
                    encoder_net=encoder_net,  # can also pass VAE model instead
                    decoder_net=decoder_net,  # of separate encoder and decoder
                    latent_dim=latent_dim,
                    samples=5)
    
    # train
    od.fit(X_train,
           loss_fn=tf.keras.losses.mse,
           epochs=5,
           verbose=True)

    # save the trained outlier detector

od.infer_threshold(X_threshold, threshold_perc=100-outlier_perc, outlier_perc=100)
print('New threshold: {}'.format(od.threshold))

save_detector(od, filepath)

od_preds = od.predict(X_outlier,
                      outlier_type='instance',
                      return_feature_score=True,
                      return_instance_score=True)

