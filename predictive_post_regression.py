import os
import shutil

import luigi
from luigi import LocalTarget
from luigi.s3 import S3Target, S3Client
from luigi.contrib.esindex import ElasticsearchTarget
from luigi.parameter import Parameter
from luigi.tools.range import RangeDailyBase
from luigi.contrib.ssh import RemoteTarget

from transfer import RemoteToS3Task, S3ToLocalTask, LocalToS3Task, LocalToRemoteTask

import pandas as pd
import numpy as np

import glob
import time
import re
import csv

from sklearn.linear_model import LinearRegression
from sklearn.grid_search import GridSearchCV

from sklearn.externals import joblib

from dateutil import relativedelta
from dateutil import parser
from datetime import datetime
from dateutil.tz import tzlocal

class LoadTrainingData(S3ToLocalTask):
  date = luigi.DateParameter()
  
  req_remote_host = luigi.Parameter(default='ubuntu@ec2-23-21-255-214.compute-1.amazonaws.com')
  req_remote_path = luigi.Parameter(default='labs/trainers/time_series_popular_tweets.csv')
  req_key_file    = luigi.Parameter(default='/Users/felipeclopes/.ec2/encore') #?????????

  s3_path     = luigi.Parameter(default='s3://encorealert-luigi-development/predictive_post_regression/raw/time_series_popular_tweets.csv')
  local_path  = luigi.Parameter(default='data/predictive_post_regression/raw/time_series_popular_tweets.csv')  

  def requires(self):
    return RemoteToS3Task(host=self.req_remote_host, 
      remote_path=self.date.strftime(self.req_remote_path + '.' + '%Y%m%d'), 
      s3_path=self.date.strftime(self.s3_path + '.' + '%Y%m%d'), 
      key_file=self.req_key_file)

  def input_target(self):
    return S3Target(self.date.strftime(self.s3_path + '.' + '%Y%m%d'), client=self._get_s3_client())

  def output_target(self):
    return LocalTarget(self.date.strftime(self.local_path + '.' + '%Y%m%d'))

class EnrichTrainingData(luigi.Task):
  date = luigi.DateParameter()
  
  input_prefix = 'data/predictive_post_regression/raw/time_series_popular_tweets.csv'
  output_prefix = 'data/predictive_post_regression/csv/enriched-time_series_popular_tweets.csv'

  token = luigi.Parameter(default='22911906-GR7LBJ2oil3cc27aUIAln4zur4F7CdKAKyEi6NDzi')
  token_key = luigi.Parameter(default='FZbyPm1i3BMfiXKlKPuzBdRlvbenW09n8LX5OvgM85g')
  con_secret = luigi.Parameter(default='cyZ6NLdySvTkhKGUGmXMKw')
  con_secret_key = luigi.Parameter(default='5UgOJOanohNPMVkfLY85CjzdMcNAAVBlRCyGYys')

  def input(self):
    return LocalTarget(self.input_file())

  def output(self):
    return [LocalTarget(self.output_file())]

  def input_file(self):
    return self.date.strftime(self.input_prefix + '.' + '%Y%m%d')

  def output_file(self):
    return self.date.strftime(self.output_prefix + '.' + '%Y%m%d')

  def requires(self):
    return LoadTrainingData(self.date)

  def run(self):
    input_file = self.date.strftime(self.input_prefix + '.' + '%Y%m%d')
    output_file = self.date.strftime(self.output_prefix + '.' + '%Y%m%d')

    print self.input_file()
    df = self.load_dataframe(self.input_file())
    df = self.enrich_dataframe(df)

    self.output()[0].makedirs()
    df.to_csv(self.output_file(), index=False)

  def load_dataframe(self, full_name):
    df_cols = map(lambda c: "C"+str(c), range(1,110))
    cols_names = ["native_id"] + df_cols

    dtype = {}
    for c in df_cols:
        dtype[c] = np.int32

    return pd.read_csv('data/time_series_popular_tweets_all_crop_100.csv', 
                       dtype=dtype, header=None, names=cols_names)

  def enrich_dataframe(self, df):
    nrows = len(df)

    # Calculate mean values for each activity
    mean_col = pd.Series(map(lambda i: (df.T)[i].values[1:].mean(), range(nrows)), index=range(nrows))

    # Calculate median values for each activity
    median_col = pd.Series(map(lambda i: np.median((df.T)[i].values[1:]), range(nrows)), index=range(nrows))

    # Calculate min values for each activity
    min_col = pd.Series(map(lambda i: (df.T)[i].values[1:].min(), range(nrows)), index=range(nrows))

    # Calculate max values for each activity
    max_col = pd.Series(map(lambda i: (df.T)[i].values[1:].max(), range(nrows)), index=range(nrows))

    # Calculate total retweets for each activity
    total_col = pd.Series(map(lambda i: sum((df.T)[i].values[1:]), range(nrows)), index=range(nrows))

    df["mean"]   = mean_col
    df["median"] = median_col
    df["min"]    = min_col
    df["max"]    = max_col
    df["total"]  = total_col

    return df

class TrainRegression(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  start        = luigi.Parameter(default=datetime(2015,07,01))
  
  s3_csvs      = luigi.Parameter('s3://encorealert-luigi-development/predictive_post_regression/csv/')
  s3_models    = luigi.Parameter('s3://encorealert-luigi-development/predictive_post_regression/models/')
  
  local_csvs   = 'data/predictive_post_regression/csv/'
  local_path   = 'data/predictive_post_regression/models/'

  def requires(self):
    yield [S3ToLocalTask(s3_path=self.s3_csvs + s3_file, local_path=self.local_csvs + s3_file) for s3_file in S3Client().list(path=self.s3_csvs)]
    yield RangeDailyBase(start=self.start, of='EnrichTrainingData')

  def output(self):
    return {
      'model': S3Target(self.model_path(self.s3_models)),
      }

  def run(self):
    time_series_retweets = self.concat_dataframes(self.local_csvs + '*')

    X = np.array(time_series_retweets.drop(["native_id","min","max","median","total"], axis=1))
    y = time_series_retweets.total.values
        
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    if not os.path.exists(self.local_path):
      os.makedirs(self.local_path)

    # Persist models for further use
    joblib.dump(lr_model, self.model_path(self.local_path), compress=9)

    with open(self.model_path(self.local_path)) as model_pickle:
      with self.output()['model'].open(mode='w') as s3_model:
        s3_model.write(model_pickle.read())

    os.remove(self.model_path(self.local_path))

  def model_path(self, directory):
    return directory + self.date.strftime('predictive_post_regression_lr_model_%Y%m%d.pkl')

  def concat_dataframes(self, wildcard):
    files = glob.glob(wildcard)
    dfs = []
    for file in files:
      print '- Parsing csv:', file
      df = pd.read_csv(file, engine='python', encoding='utf-8')
      dfs.append(df)
      print '# Loaded', file, 'with', len(df), 'lines.'
    full = pd.concat(dfs)
    return full

class DeployModel(luigi.Task):
  date = luigi.Parameter(default=datetime.today())
  
  model_local_directory = 'data/predictive_post_regression/deploy/'

  def requires(self):
    return TrainRegression(self.date)

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
