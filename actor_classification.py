import os
import re
import fnmatch
import json
import nltk

import numpy as np
import pandas as pd
import chardet
import gc
import matplotlib.pyplot as plt

from pprint import pprint
from time import time, ctime
from datetime import datetime
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import glob
import csv
import luigi

from boto.s3.connection import S3Connection

from luigi import LocalTarget
from luigi.s3 import S3Target, S3Client
from luigi.parameter import Parameter
from luigi.tools.range import RangeDailyBase
from luigi.contrib.ssh import RemoteTarget

from transfer import RemoteToS3Task, S3ToLocalTask, LocalToS3Task, LocalToRemoteTask

################################################
#### SCIKIT-LEARN TRANSFORMATORS
################################################

class VerifiedTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X.verified.fillna(False, inplace=True)
    X.verified = LabelEncoder().fit_transform(X.verified)
    return X


class LangOneHotEncoding(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    valid_langs = list(set(X.lang) - set([None, np.nan, 'Select Language...']))
    self.feature_names_ = ["lang_"+str(l) for l in valid_langs if type(l) == str]
    return self

  def transform(self, X, y=None):
    check_is_fitted(self, 'feature_names_')
    
    X["lang"].fillna("", inplace=True)
    for lang_feature in self.feature_names_:
        X[lang_feature] = [(1 if lang_feature == "lang_"+v else 0) for v in X["lang"].values]
    
    X.drop(["lang"], axis=1, inplace=True)
    return X
    

class FillTextNA(BaseEstimator, TransformerMixin):

  def __init__(self, cols, replace_by=""):
    self.cols = cols
    self.replace_by = replace_by

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
        if c in X:
            X[c].fillna(self.replace_by, inplace=True)
    return X


class DataFrameTfidfVectorizer(TfidfVectorizer):

  def __init__(self, col, prefix=None, input='content', encoding='utf-8',
               decode_error='strict', strip_accents=None, lowercase=True,
               preprocessor=None, tokenizer=None, analyzer='word',
               stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
               ngram_range=(1, 1), max_df=1.0, min_df=1,
               max_features=None, vocabulary=None, binary=False,
               dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
               sublinear_tf=False):
      super(DataFrameTfidfVectorizer, self).__init__(
          input=input, encoding=encoding, decode_error=decode_error,
          strip_accents=strip_accents, lowercase=lowercase,
          preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
          stop_words=stop_words, token_pattern=token_pattern,
          ngram_range=ngram_range, max_df=max_df, min_df=min_df,
          max_features=max_features, vocabulary=vocabulary, binary=binary,
          dtype=dtype)

      self.col = col
      self.prefix = prefix or col
      
  def treat_special_char(self, c):
    try:
      encoding = chardet.detect(str(c))['encoding'] or "KOI8-R"
      return '0' if c.isdigit() else c.decode(encoding)
    except:        
      return '9'

  def treat_special_chars(self, col):
    col.fillna("null", inplace=True)
    col = [''.join([self.treat_special_char(c) for c in list(n)]) 
           for n in col.values]
    return col

  def fit(self, dataframe, y=None):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    super(DataFrameTfidfVectorizer, self).fit(dataframe[self.col])
    return self

  def fit_transform(self, dataframe, y=None):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    field_matrix = super(DataFrameTfidfVectorizer, self).fit_transform(dataframe[self.col])
    features_names = map(lambda f: "_".join([self.prefix,f]), super(DataFrameTfidfVectorizer, self).get_feature_names())
    field_df = pd.DataFrame(field_matrix.A, columns=features_names)

    dataframe = dataframe.join(field_df)

    return dataframe

  def transform(self, dataframe, copy=True):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    field_matrix = super(DataFrameTfidfVectorizer, self).transform(dataframe[self.col])
    features_names = map(lambda f: "_".join([self.prefix,f]), super(DataFrameTfidfVectorizer, self).get_feature_names())
    field_df = pd.DataFrame(field_matrix.A, columns=features_names)

    dataframe = dataframe.join(field_df)

    return dataframe


class TextToLowerCase(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X[c] = [t.lower() for t in X[c].values]
    return X


class NumberOfWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_words_in_"+c] = [len(t.split(' ')) for t in X[c].values]
    return X


class NumberNonAlphaNumChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_non_alphanum_in_"+c] = [len(re.sub(r"[\w\d]","", t)) for t in X[c].values]
    return X


class NumberUpperCaseChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_upper_case_chars_in_"+c] = [len(re.sub(r"[^A-Z]","", t)) for t in X[c].values]
    return X


class NumberCamelCaseWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_camel_case_words_in_"+c] = [len(re.findall(r"^[A-Z][a-z]|\s[A-Z][a-z]", t)) 
                                                 for t in X[c].values]
    return X


class NumberOfMentions(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_mentions_in_"+c] = [len(re.findall(r"\s@[a-zA-Z]",t)) 
                                                 for t in X[c].values]
    return X


class NumberOfPeriods(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_periods_in_"+c] = [len(t.split(". ")) 
                                        for t in X[c].values]
    return X


class AvgWordsPerPeriod(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["avg_words_per_period_in_"+c] = [np.mean([len(p.split(" ")) for p in t.split(". ")]) 
                                            for t in X[c].values]
    return X


class MentionToFamilyRelation(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for tkn in tokenizer.tokenize(t):
          if tkn in ["husband","wife","father","mother","daddy","mommy",
                     "grandfather","grandmother","grandpa","grandma"]:
                count += 1
    return count

  def transform(self, X, y=None):
    for c in self.cols:
        if c in X:
            X["mention_to_family_relation_in_"+c] = [self.count_mentions(t) 
                                                     for t in X[c].values]
    return X


class MentionToOccupation(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    occupations = pd.read_csv("https://raw.githubusercontent.com/johnlsheridan/occupations/master/occupations.csv")
    occupations = [o.lower().split(' ')[-1] for o in occupations.Occupations.values]
    self.occupations_ = dict.fromkeys(occupations, 1)
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for name in tokenizer.tokenize(t):
        count += self.occupations_.get(name, 0)
    return count

  def transform(self, X, y=None):
    check_is_fitted(self, 'occupations_')
    for c in self.cols:
        if c in X:
            X["mention_to_occupation_in_" + c] = [self.count_mentions(t) 
                                                 for t in X[c].values]
    return X
    

class PersonNames(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    female_names = pd.read_csv("http://deron.meranda.us/data/census-dist-female-first.txt", names=["name"])
    male_names   = pd.read_csv("http://deron.meranda.us/data/census-dist-male-first.txt", names=["name"])
    female_names = [re.sub(r"[^a-z]","",n.lower()) for n in female_names.name.values]
    male_names   = [re.sub(r"[^a-z]","",n.lower()) for n in male_names.name.values]        
    self.person_names_ = dict.fromkeys(set(male_names + female_names), 1)
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for name in tokenizer.tokenize(t):
        count += self.person_names_.get(name, 0)
    return count

  def transform(self, X, y=None):
    check_is_fitted(self, 'person_names_')
    for c in self.cols:
        if c in X:
            X["person_names_in_" + c] = [self.count_mentions(t) 
                                        for t in X[c].values]
    return X   

class DropColumnsTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.copy()
    for c in self.cols:
      if c in X:
        X.drop([c], axis=1, inplace=True)
    return X


class NumpyArrayTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.copy()
    X = X.reindex_axis(sorted(X.columns), axis=1)
    X.fillna(0, inplace=True)
    return np.asarray(X)


class Debugger(BaseEstimator, TransformerMixin):
  def __init__(self, name=""):
      self.name = name

  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None):
      print self.name, '-', ctime(), X.shape
      return X

################################################
#### TRAIN MODEL
################################################
class TrainModel(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  str_today    = datetime.today().strftime('%Y%m%d')

  bucket       = luigi.Parameter('encore-luigi-development')
  bucket_dir   = luigi.Parameter('actor_classification/raw/')
  bucket_file  = luigi.Parameter('actor_classification_train.csv')
  
  input_dir    = luigi.Parameter('./data/actor_classification/raw/')
  output_dir   = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/models/')
  local_path   = './data/actor_classification/models/'

  def model_path(self, directory):
    return directory + self.date.strftime('actor_classification_trained_model_%Y%m%d.pkl')

  def load_input_dataframe(self):
    train = None
    for file in os.listdir(self.input_dir):
      if fnmatch.fnmatch(file, self.bucket_file + '.*') and os.stat(self.input_dir+file).st_size > 0:
        if train is None:
          print "==> Initializing input dataframe with " + file + ": " + self.str_today
          train = pd.read_csv(open(self.input_dir+file,'rU'),
                              engine='python', sep=",", quoting=1)
        else:
          print "==> Concatenating dataframe with " + file + ": " + self.str_today
          train = pd.concat([train, pd.read_csv(open(self.input_dir+file,'rU'),
                              engine='python', sep=",", quoting=1)])
        train.drop_duplicates(inplace=True)
        print train.shape
    return train

  def output(self):
    return S3Target(self.model_path(self.output_dir))

  def requires(self):
    conn    = S3Connection()
    bucket  = conn.get_bucket(self.bucket)
    files   = bucket.list(self.bucket_dir + self.bucket_file)
    for s3_file in files:
      s3_file_name = s3_file.name.split('/')[-1]
      yield S3ToLocalTask(input_path_s3='s3://' + self.bucket + '/' + self.bucket_dir + s3_file_name, output_path_local=self.input_dir + s3_file_name)

  def run(self):
    # Read input dataset
    print "==> Loading raw data: " + self.str_today
    train = self.load_input_dataframe()

    print "==> Performing a K-fold CV: " + self.str_today

    outcome = "manual_segment"

    features = list(set(train.columns) - set([outcome]))

    n_estimators = 100

    # Model Pipeline
    pipeline = Pipeline([ ("drop_cols", DropColumnsTransformer(["segment","link"])),
                      ("verified", VerifiedTransformer()),
                      ("lang", LangOneHotEncoding()),
                      ("fill_text_na", FillTextNA(["screen_name","name","summary"], "null")),
                      ("debugger1", Debugger('Starting')),
                      ("qt_words", NumberOfWords(["name","summary"])),
                      ("qt_non_alphanum_chars", NumberNonAlphaNumChars(["name","summary"])),
                      ("qt_upper_case_chars", NumberUpperCaseChars(["name","summary"])),
                      ("qt_camel_case_words", NumberCamelCaseWords(["name","summary"])),
                      ("qt_mentions", NumberOfMentions(["summary"])),
                      ("qt_periods", NumberOfPeriods(["summary"])),
                      ("avg_words_per_period", AvgWordsPerPeriod(["summary"])),
                      ("lower_case", TextToLowerCase(["screen_name","name","summary"])),
                      ("family", MentionToFamilyRelation(["summary"])),
                      ("debugger2", Debugger('Basic Statistics')),
                      ("person_names", PersonNames(["name"])),
                      ("debugger3", Debugger('Person Names')),
                      ("occupations", MentionToOccupation(["summary"])),
                      ("debugger4", Debugger('Occupations')),
                      ("name_chars_tfidf", DataFrameTfidfVectorizer(col="name", 
                                            prefix="name_c",
                                            ngram_range=(3, 5), 
                                            analyzer="char",
                                            binary=True, #False
                                            min_df = 50,
                                            max_features=50)),
                      ("name_words_tfidf", DataFrameTfidfVectorizer(col="name", 
                                            prefix="name_w", 
                                            token_pattern=r'\w+',
                                            ngram_range=(1, 2), 
                                            analyzer="word",
                                            binary=True, #False
                                            min_df = 10,
                                            max_features=50)),
                      ("debugger5", Debugger('Names TFIDF')),
                      ("screen_name_tfidf", DataFrameTfidfVectorizer(col="screen_name", 
                                            ngram_range=(3, 5), 
                                            analyzer="char",
                                            binary=True, #False
                                            min_df = 50,
                                            max_features=50)),
                      ("debugger6", Debugger('Screen Names TFIDF')),
                      ("summary_tfidf", DataFrameTfidfVectorizer(col="summary",
                                          token_pattern=r'\w+',
                                          ngram_range=(1, 3), 
                                          analyzer="word",
                                          binary=True, #False
                                          sublinear_tf=True, 
                                          stop_words='english',
                                          min_df = 50,
                                          max_features=50)),
                      ("debugger7", Debugger('Summary TFIDF')),
                      ("drop_text_cols", DropColumnsTransformer(["screen_name","name","summary"])),
                      ("nparray", NumpyArrayTransformer()),
                      ("debugger8", Debugger('Finish')),
                      ("model", RandomForestClassifier())])

    k_fold = KFold(n=len(train), n_folds=2, shuffle=True)
    b_scores, svc_scores = [], []

    for tr_indices, cv_indices in k_fold:
      tr    = train.iloc[tr_indices,:].loc[:, features].copy()
      cv    = train.iloc[cv_indices,:].loc[:, features].copy()

      tr_y  = train.iloc[tr_indices,:][outcome].values
      cv_y  = train.iloc[cv_indices,:][outcome].values

      pipeline.fit(tr, tr_y)

      print(confusion_matrix(cv_y, pipeline.predict(cv)))    
      print('#### SCORE:' + str(pipeline.score(cv, cv_y)))

    print "==> Training model: " + self.str_today

    pipeline.set_params(model__n_estimators = n_estimators)
    pipeline.fit(train.loc[:,features], train.loc[:,outcome])

    if not os.path.exists(self.local_path):
      os.makedirs(self.local_path)

    print '==> Persisting model with pickle - ' + self.str_today

    joblib.dump(pipeline, self.model_path(self.local_path), compress=9)

    with open(self.model_path(self.local_path)) as model_pickle:
      with self.output().open(mode='w') as s3_model:
        s3_model.write(model_pickle.read())

    os.remove(self.model_path(self.local_path))

    print '==> Pickle model persisted - ' + self.str_today


################################################
#### DEPLOY MODEL
################################################
class DeployModel(luigi.Task):
  date = luigi.Parameter(default=datetime.today())
  
  output_dir = luigi.Parameter('./data/actor_classification/deploy/')

  def requires(self):
    return TrainModel(self.date)

  def output(self):
    return LocalTarget(self.model_path(self.input().path, self.output_dir))

  def run(self):
    print "S3ToLocalTask -", 'input_path_s3:', self.input().path, 'output_path_local:', self.model_path(self.input().path, self.output_dir)
    S3ToLocalTask(input_path_s3=self.input().path, output_path_local=self.model_path(self.input().path, self.output_dir)).run()

  def model_path(self, path, directory):
    filename = path.split('/')[-1]
    return directory + filename

if __name__ == "__main__":
    luigi.run()