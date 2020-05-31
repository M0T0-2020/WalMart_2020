import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
import lightgbm as lgb
from time import time
from tqdm import tqdm
from collections import Counter
from scipy import stats
import gc, pickle
import ast

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold,TimeSeriesSplit, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import minmax_scale