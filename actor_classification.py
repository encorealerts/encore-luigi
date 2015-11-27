import os
import fnmatch
import gc
import sys
import shutil
import chardet

import glob
import time
import re
import csv

from datetime import datetime

import luigi
from luigi import LocalTarget
from luigi.s3 import S3Target, S3Client
from luigi.parameter import Parameter
from luigi.tools.range import RangeDailyBase
from luigi.contrib.ssh import RemoteTarget

from transfer import RemoteToS3Task, S3ToLocalTask, LocalToS3Task, LocalToRemoteTask

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

################################################
#### DOWNLOAD TRAINING DATA FROM S3
################################################
class DownloadTrainingData(S3ToLocalTask):
  date = luigi.DateParameter()
  
  req_remote_host = luigi.Parameter(default='ubuntu@ec2-23-21-255-214.compute-1.amazonaws.com')
  req_remote_path = luigi.Parameter(default='labs/trainers/actor_classification_train.csv')
  req_key_file    = luigi.Parameter(default='/Users/felipeclopes/.ec2/encore')

  s3_path     = luigi.Parameter(default='s3://encorealert-luigi-development/actor_classification/raw/actor_classification_train.csv')
  local_path  = luigi.Parameter(default='/mnt/encore-luigi/data/actor_classification/raw/actor_classification_train.csv')  

  def requires(self):
    return RemoteToS3Task(host=self.req_remote_host, 
      remote_path=self.date.strftime(self.req_remote_path + '.' + '%Y%m%d'), 
      s3_path=self.date.strftime(self.s3_path + '.' + '%Y%m%d'), 
      key_file=self.req_key_file)

  def input_target(self):
    return S3Target(self.date.strftime(self.s3_path + '.' + '%Y%m%d'), client=self._get_s3_client())

  def output_target(self):
    return LocalTarget(self.date.strftime(self.local_path + '.' + '%Y%m%d'))


################################################
#### PREPROCESS DATA
################################################
class PreprocessData(luigi.Task):
  str_today     = datetime.today().strftime('%Y%m%d')
  start_date    = datetime(2015,11,24)
  
  input_dir     = '/mnt/encore-luigi/data/actor_classification/raw/'
  input_prefix  = 'actor_classification_train.csv'
  output_prefix = '/mnt/encore-luigi/data/actor_classification/csv/enriched-actor_classification_train.csv'
  output_file   = output_prefix + '.' + str_today

  def output(self):
    return LocalTarget(self.output_file)

  def requires(self):
    yield RangeDailyBase(start=self.start_date, of='DownloadTrainingData')

  def load_input_dataframe(self):
    train = None
    for file in os.listdir(self.input_dir):
      if fnmatch.fnmatch(file, self.input_prefix+'.*') and os.stat(self.input_dir+file).st_size > 0:
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

  def save_output_file(self, df):
    self.output().makedirs()
    df.to_csv(self.output_file, index=False)

  def run(self):
    # Read input dataset
    print "==> Loading raw data: " + self.str_today
    train = self.load_input_dataframe()

    # Perform feature engineering
    train = self.perform_feature_engineering(train)

    # Save output file
    print "==> Persisting preprocessed data: " + self.output_file
    self.save_output_file(train)

  def perform_feature_engineering(self, train):
    print "==> Feature Engineering - Remove non-relevant columns: " + self.str_today
    del train["segment"]
    del train["link"]

    print "==> Feature Engineering - Transform boolean 'verified' to 0/1: " + self.str_today
    train.ix[train.verified.isnull(), 'verified'] = False
    train.ix[train.verified == True,  'verified'] = 1
    train.ix[train.verified == False, 'verified'] = 0

    print "==> Feature Engineering - OneHotEncoding for 'lang': " + self.str_today
    if "lang" in train:
      train.ix[(train.lang == 'Select Language...') | (train.lang.isnull()), 'lang'] = None
      for lang in list(set(train.lang)):
        if lang != None:
          train.ix[train.lang == lang, "lang_"+lang] = 1
          train.ix[train.lang != lang, "lang_"+lang] = 0
      del train["lang"]

    gc.collect()

    print "==> Feature Engineering - Treat special characters: " + self.str_today
    text_fields = ["name", "screen_name","summary"]

    def treat_special_char(c):
      try:
        encoding = chardet.detect(str(c))['encoding'] or "KOI8-R"
        return '0' if c.isdigit() else c.decode(encoding)
      except UnicodeDecodeError:        
        return '9'

    for field in text_fields:
      train.ix[train[field].isnull(), field] = "null"
      train[field] = map(lambda n: ''.join(map(lambda c: treat_special_char(c), list(n))), train[field].values)

    def num_char_tokenizer(text):
      return list(text)

    for field in ["screen_name","name"]:
      if field in train:
        print "==> Feature Engineering - TfidfVectorizer for '"+field+"': " + self.str_today
        field_tfidf = TfidfVectorizer(tokenizer=num_char_tokenizer,
                                      ngram_range=(3, 5), 
                                      analyzer="char",
                                      min_df = 50) # 8

        print "==> Feature Engineering - TfidfVectorizer for '"+field+"' - fit_transform: " + self.str_today
        field_matrix = field_tfidf.fit_transform(train[field])
        features_names = map(lambda f: "_".join([field,f]), field_tfidf.get_feature_names())
        print "==> Feature Engineering - TfidfVectorizer for '"+field+"' - data frame: " + self.str_today
        field_df = pd.DataFrame(field_matrix.A, columns=features_names)

        print "==> Feature Engineering - TfidfVectorizer for '"+field+"' - concat: " + self.str_today
        train = pd.concat([train, field_df], axis=1, join='inner')
        gc.collect()

        print "==> Feature Engineering - TfidfVectorizer for '"+field+"' - drop: " + self.str_today
        del train[field]
        gc.collect()
        print "==> Feature Engineering - TfidfVectorizer for '"+field+"' - dropped: " + self.str_today

    if "summary" in train:
      print "==> Feature Engineering - TfidfVectorizer for 'summary': " + self.str_today
      summary_tfidf = TfidfVectorizer(token_pattern=r'\w+',
                                      ngram_range=(1, 4), 
                                      analyzer="word",
                                      min_df = 50) #5

      print "==> Feature Engineering - TfidfVectorizer for 'summary' - fit_transform: " + self.str_today
      summary_matrix = summary_tfidf.fit_transform(train.summary)
      features_names = map(lambda f: "_".join(["summary",f]), summary_tfidf.get_feature_names())
      print "==> Feature Engineering - TfidfVectorizer for 'summary' - data_frame: " + self.str_today
      summary_df = pd.DataFrame(summary_matrix.A, columns=features_names)
      print "==> Feature Engineering - TfidfVectorizer for 'summary' - concat: " + self.str_today
      train = pd.concat([train, summary_df], axis=1, join='inner')
      gc.collect()
      print "==> Feature Engineering - TfidfVectorizer for 'summary' - drop: " + self.str_today
      del train["summary"]
      gc.collect()
      print "==> Feature Engineering - TfidfVectorizer for 'summary' - dropped: " + self.str_today

    print "==> Feature Engineering - Treat remaining null values: " + self.str_today
    train.fillna(0, inplace=True)
    gc.collect()

    return train
    

################################################
#### TRAIN A RANDOM FOREST MODEL
################################################
class TrainRandomForestModel(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  
  s3_csvs      = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/csv/')
  s3_models    = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/models/')
  
  local_csvs   = '/mnt/encore-luigi/data/actor_classification/csv/'
  local_path   = '/mnt/encore-luigi/data/actor_classification/models/'

  input_prefix = '/mnt/encore-luigi/data/actor_classification/csv/enriched-actor_classification_train.csv'

  def input_file(self):
    return self.date.strftime(self.input_prefix + '.' + '%Y%m%d')

  def requires(self):
    yield [S3ToLocalTask(s3_path=self.s3_csvs + s3_file, local_path=self.local_csvs + s3_file) for s3_file in S3Client().list(path=self.s3_csvs)]
    yield PreprocessData()

  def output(self):
    return {
      'model': S3Target(self.model_path(self.s3_models)),
      'model_features': S3Target(self.model_features_path(self.s3_models))
      }

  def run(self):
    train = pd.read_csv(self.input_file())
    print('Loaded:' + self.input_file())

    outcome = "manual_segment"

    features = list(set(train.columns) - set([outcome]))

    k_fold = KFold(n=len(train), n_folds=4, indices=False, shuffle=True)
    b_scores, svc_scores = [], []

    print('==> Starting K-fold CV for ' + self.input_file())
    for tr_indices, cv_indices in k_fold:
        tr   = np.asarray(train[tr_indices][features])
        tr_y = np.asarray(train[tr_indices][outcome])

        cv   = np.asarray(train[cv_indices][features])
        cv_y = np.asarray(train[cv_indices][outcome])

        rfmodel = RandomForestClassifier(n_estimators=25)
        rfmodel.fit(tr, tr_y)

        print(confusion_matrix(cv_y, rfmodel.predict(cv)))    
        print('score:' + str(rfmodel.score(cv, cv_y)))
        
    rfmodel = RandomForestClassifier(n_estimators=25)
    rfmodel.fit(train[features], train[outcome])

    if not os.path.exists(self.local_path):
      os.makedirs(self.local_path)

    print('==> Persisting pickle files for ' + self.input_file())

    joblib.dump(rfmodel, self.model_path(self.local_path), compress=9)
    joblib.dump(train.columns, self.model_features_path(self.local_path), compress=9)

    with open(self.model_path(self.local_path)) as model_pickle:
      with open(self.model_features_path(self.local_path)) as model_features_pickle:
        with self.output()['model'].open(mode='w') as s3_model:
          with self.output()['model_features'].open(mode='w') as s3_model_features:
            s3_model.write(model_pickle.read())
            s3_model_features.write(model_features_pickle.read())

    os.remove(self.model_path(self.local_path))
    os.remove(self.model_features_path(self.local_path))

    print('==> Pickle files persisted for ' + self.input_file())

  def model_path(self, directory):
    return directory + self.date.strftime('actor_classification_random_forest_%Y%m%d.pkl')

  def model_features_path(self, directory):
    return directory + self.date.strftime('actor_classification_random_forest_features_%Y%m%d.pkl')

################################################
#### DEPLOY MODEL
################################################
class DeployModel(luigi.Task):
  date = luigi.Parameter(default=datetime.today())
  
  model_local_directory = '/mnt/encore-luigi/data/actor_classification/deploy/'

  def requires(self):
    return TrainRandomForestModel(self.date)

  def output(self):
    for name in self.input():
      s3_input = self.input()[name]
      return LocalTarget(self.model_path(s3_input.path, self.model_local_directory))

  def run(self):
    for name in self.input():
      s3_input = self.input()[name]
      S3ToLocalTask(s3_path=s3_input.path, local_path=self.model_path(s3_input.path, self.model_local_directory)).run()

  def model_path(self, path, directory):
    filename = path.split('/')[-1]
    return directory + filename

if __name__ == "__main__":
    luigi.run()