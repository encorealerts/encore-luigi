import os
import sys
import shutil

import luigi
from luigi import LocalTarget
from luigi.s3 import S3Target
from luigi.contrib.esindex import ElasticsearchTarget
from luigi.parameter import Parameter
from luigi.tools.range import RangeDailyBase
from luigi.contrib.ssh import RemoteTarget

import elasticsearch
from elasticsearch import Elasticsearch
from transfer import S3ToLocalTask

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

class TrainRegressionModel(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  start        = luigi.Parameter(default=datetime(2015,07,01))
  
  s3_models    = luigi.Parameter('s3://encorealert-luigi-development/predictive_post_regression/models/')
  local_path   = 'data/predictive_post_regression/models/'

  es = Elasticsearch()

  START_DATE = datetime.now() - timedelta(days=120)

  def output(self):
    return {
      'model': S3Target(self.model_path(self.s3_models)),
      }

  def run(self):
    time_series_retweets = self.enrich_data(self.load_data())

    y = time_series_retweets.total.values
    X = np.array(time_series_retweets.drop(["native_id","min","max","median","total"], axis=1))
        
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

  def load_data(self):
    df_cols = map(lambda c: "C"+str(c), range(1,110))
    cols_names = ["native_id"] + df_cols

    # dtype = {}
    # for c in df_cols:
    #     dtype[c] = np.int32

    df = pd.DataFrame(columns=cols_names)#, dtype=dtype)

    date_from = self.START_DATE

    # TODO Load from ElasticSearch
    for f in range(200):
      try:
        ts = self.time_series_for_retweets(self.most_retweeted_tweets(date_from + timedelta(days=f)))

        for tweet in ts.keys():
          new_row = {"native_id": tweet}

          for i in range(1,110):
            try:
              new_row = {"C{0}".format(i): ts[tweet][i]}
            except:
              new_row = {"C{0}".format(i): 0}

          df.append(new_row)
      except:
        msg = str(sys.exc_info()[0])
        if not "elasticsearch.exceptions.NotFoundError" in msg:
          print "An error occurred while trying to read from ElasticSearch: " + str(sys.exc_info()[0])

    return df


  def enrich_data(self, df):

    nrows = len(df)
    idx = range(nrows)

    time_series_for_row = lambda i: (df.T)[i].values[1:]

    mean_col   = pd.Series(map(lambda i: time_series_for_row(i).mean(), idx), index=idx)
    median_col = pd.Series(map(lambda i: np.median(time_series_for_row(i)), idx), index=idx)
    min_col    = pd.Series(map(lambda i: time_series_for_row(i).min(), idx), index=idx)
    max_col    = pd.Series(map(lambda i: time_series_for_row(i).max(), idx), index=idx)
    total_col  = pd.Series(map(lambda i: sum(time_series_for_row(i)), idx), index=idx)

    df["mean"]   = mean_col
    df["median"] = median_col
    df["min"]    = min_col
    df["max"]    = max_col
    df["total"]  = total_col

    return df

  def most_retweeted_tweets(self, date_from, n_tweets=10, threshold = 500):

    date_from = date_from.date()
    to   = date_from + timedelta(days=1)

    # Elastic Search query
    body_search = {
      "from": 0,
      "size": n_tweets,
      "query": {
        "bool": {
          "must": [
            { "range": { "created_at": { "from": date_from, "to": to } } }
          ]
        }
      },
      "aggs": {
        "retweeted_activities": {
          "terms": { "field": "shared_native_id"}
        }
      }
    }

    # Query execution
    search = self.es.search(
      index = self.activities_indexes(date_from, to),
      search_type = 'count',
      body  = body_search
    )

    # Query return map for relevant values
    most_retweeted = filter(lambda b: b["doc_count"] > threshold, search['aggregations']['retweeted_activities']['buckets'])

    native_ids = map(lambda e: e["key"],most_retweeted)

    # for native_id in native_ids:
    #   print "####### native_ids (begin) ########"
    #   print "Index: " + str(self.activities_indexes(date_from, to))
    #   print native_ids

    return native_ids

  def time_series_for_retweets(self, tweets_ids):

    time_series_per_popular_tweet = {}

    passed = 0

    for tweet_id in tweets_ids:
      try:
        created_at_str = None
        # Obtain created_at
        #tweet_id = "635954756800786433"
        for w in range(1,54):
          try:
            body_search = { "query": { "bool": { "must": [ { "term": { "native_id": tweet_id } } ] } } }
            search = self.es.search(
              index = "activities_week_{0}".format(w),
              body  = body_search
            )

            created_at_str = search["hits"]["hits"][0]["_source"]["created_at"] # ex: "2015-06-22T23:27:06.000Z"

            break
          except:
            None

        if created_at_str == None:
          continue

        passed += 1
        print "####### created_at (begin) ########"
        print passed
        print created_at_str
        print "####### created_at (end) ########"

        date_from  = datetime.strptime(created_at_str[:10], '%Y-%m-%d')
        to    = date_from + timedelta(hours=1)

        # Elastic Search query
        body_search = {
          "query": {
            "bool": {
              "must": [
                { "range": { "created_at": { "from": date_from, "to": to } } },
                { "term": { "verb": 'share' } },
                { "term": { "shared_native_id": tweet_id } }
              ]
            }
          },
          "aggs": {
            "activities_per_timeframe": {
              "date_histogram": {
                "field": "created_at",
                "interval": "5s",
                "format": "yyyy-MM-dd HH:mm:ss",
                "min_doc_count": 0
              }
            }
          }
        }

        # Query execution
        search = self.es.search(
          index = self.activities_indexes(date_from, to),
          search_type = 'count',
          body = body_search
        )

        # Query return map for relevant values
        search = map(lambda b: b["doc_count"], search['aggregations']['activities_per_timeframe']['buckets'])

        sum_counts = sum(search)
        if sum_counts is not None and sum_counts > 0:
          time_series_per_popular_tweet[tweet_id] = search
      except Exception as e:
        if type(e) != elasticsearch.exceptions.NotFoundError:
          print type(e)
          print e.args
          print "Error with native_id: " + tweet_id
          raise

    return time_series_per_popular_tweet

  def activities_indexes(self, date_from, to=None):
    a = date_from.isocalendar()[1]
    b = a if to is None else to.isocalendar()[1]
    week_label = lambda w: "activities_week_{0}".format(w)
    if a <= b:
      indices  = map(week_label, range(a,b+1))
    else:
      indices  = map(week_label, range(a, 53))
      indices += map(week_label, range(1,b+1))

    return indices

class DeployModel(luigi.Task):
  date = luigi.Parameter(default=datetime.today())
  
  model_local_directory = 'data/predictive_post_regression/deploy/'

  def requires(self):
    return TrainRegressionModel(self.date)

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
