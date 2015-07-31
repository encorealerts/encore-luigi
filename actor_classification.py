import os
import shutil

import luigi
from luigi import LocalTarget
from luigi.s3 import S3Target, S3Client
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

from twitter import *

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from dateutil import relativedelta
from dateutil import parser
from datetime import datetime
from dateutil.tz import tzlocal

class DownloadTrainerData(S3ToLocalTask):
  date = luigi.DateParameter()
  
  req_remote_host = luigi.Parameter(default='ubuntu@ec2-23-21-255-214.compute-1.amazonaws.com')
  req_remote_path = luigi.Parameter(default='labs/trainers/twitter-actor.csv')
  req_key_file    = luigi.Parameter(default='/Users/felipeclopes/.ec2/encore')

  s3_path     = luigi.Parameter(default='s3://encorealert-luigi-development/actor_classification/raw/twitter-actor.csv')
  local_path  = luigi.Parameter(default='data/actor_classification/raw/twitter-actor.csv')  

  def requires(self):
    return RemoteToS3Task(host=self.req_remote_host, 
      remote_path=self.date.strftime(self.req_remote_path + '.' + '%Y%m%d'), 
      s3_path=self.date.strftime(self.s3_path + '.' + '%Y%m%d'), 
      key_file=self.req_key_file)

  def input_target(self):
    return S3Target(self.date.strftime(self.s3_path + '.' + '%Y%m%d'), client=self._get_s3_client())

  def output_target(self):
    return LocalTarget(self.date.strftime(self.local_path + '.' + '%Y%m%d'))

class EnrichTrainerData(luigi.Task):
  date = luigi.DateParameter()
  input_prefix = luigi.Parameter(default='data/actor_classification/raw/twitter-actor.csv')
  output_prefix = luigi.Parameter(default='data/actor_classification/csv/enriched-twitter-actor.csv')

  token = luigi.Parameter(default='22911906-GR7LBJ2oil3cc27aUIAln4zur4F7CdKAKyEi6NDzi')
  token_key = luigi.Parameter(default='FZbyPm1i3BMfiXKlKPuzBdRlvbenW09n8LX5OvgM85g')
  con_secret = luigi.Parameter(default='cyZ6NLdySvTkhKGUGmXMKw')
  con_secret_key = luigi.Parameter(default='5UgOJOanohNPMVkfLY85CjzdMcNAAVBlRCyGYys')

  def input(self):
    return LocalTarget(self.input_file())

  def output(self):
    return [LocalTarget(self.output_file() + '.brand'), LocalTarget(self.output_file() + '.person')]

  def input_file(self):
    return self.date.strftime(self.input_prefix + '.' + '%Y%m%d')

  def output_file(self):
    return self.date.strftime(self.output_prefix + '.' + '%Y%m%d')

  def requires(self):
    return DownloadTrainerData(self.date)

  def run(self):
    input_file = self.date.strftime(self.input_prefix + '.' + '%Y%m%d')
    output_file = self.date.strftime(self.output_prefix + '.' + '%Y%m%d')

    print(self.input_file())
    df = self.load_dataframe(self.input_file())

    business_data = self.lookup_twitter_statuses(df[df['class'] == 'business'])
    self.persist_complete_data(self.output_file() + '.brand', business_data)

    personal_data = self.lookup_twitter_statuses(df[df['class'] == 'personal'])
    self.persist_complete_data(self.output_file() + '.person', personal_data)

  def load_dataframe(self, full_name):
    df = pd.read_csv(full_name, header=None, names=['native_id', 'class'])
    df = df[df['class'].isin(['business', 'personal'])]
    df = df.drop_duplicates('native_id')
    df['native_id'] = df['native_id'].astype('str')
    print '# Parsed', full_name, 'with', len(df), 'lines.'
    return df

  def persist_complete_data(self, file_name, data):
    s = 0
    with open(file_name, 'w') as csv_file:
      tweets_writer = csv.writer(csv_file)
      tweets_writer.writerow([
        'actor_id',
        'actor_screen_name',
        'actor_name',
        'actor_verified',
        'actor_friends_count',
        'actor_followers_count',
        'actor_listed_count',
        'actor_statuses_count',
        'actor_favorites_count',
        'actor_summary',
        'actor_created_at',
        'actor_location',

        'tweet_id',
        'tweet_created_at',
        'tweet_generator',
        'tweet_body',
        'tweet_verb',

        'tweet_urls_count',
        'tweet_mentions_count',
        'tweet_hashtags_count',
        'tweet_trends_count',
        'tweet_symbols_count'])
      for tweet in data:
        tweets_writer.writerow([
          tweet['user']['id'],
          tweet['user']['screen_name'],
          tweet['user']['name'],
          tweet['user']['verified'],
          tweet['user']['friends_count'],
          tweet['user']['followers_count'],
          tweet['user']['listed_count'],
          tweet['user']['statuses_count'],
          tweet['user']['favourites_count'],
          tweet['user']['description'],
          tweet['user']['created_at'],
          tweet['user']['location'] if tweet['user'].get('location') else 'null',

          tweet['id'],
          tweet['created_at'],
          re.findall('>(.*)<', tweet['source'])[0],
          tweet['text'],
          not tweet['retweeted'],
          len(tweet['entities']['urls']),
          len(tweet['entities']['user_mentions']),
          len(tweet['entities']['hashtags']),
          "",
          len(tweet['entities']['symbols'])
        ])
        s += 1
        if (s % 100 == 0): print('Written', s, 'rows to', file_name)

  def lookup_twitter_statuses(self, data):
    if len(data) == 0: return []

    t = Twitter(auth=OAuth(self.token, self.token_key, self.con_secret, self.con_secret_key))
    # Get trained_data buckets too lookup
    indices = np.arange(len(data))
    max_mod = int((len(data)/100)+1)
    tweets = []
    for x in range(0, max_mod):
      native_ids = data[(indices % max_mod) == x]['native_id']
      tweets += t.statuses.lookup(_id=','.join(native_ids), _timeout=3)
      print('Downloaded tweet info for', len(tweets))
      time.sleep(5)

    return tweets

class TrainRandomForestModel(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  start        = luigi.Parameter(default=datetime(2015,07,01))
  s3_folder    = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/csv/')
  
  local_folder = 'data/actor_classification/csv/'
  directory    = 'data/actor_classification/models/'

  tweets_features_cols = ['actor_favorites_count', 
                          'actor_followers_count', 
                          'actor_friends_count', 
                          'actor_listed_count', 
                          'actor_statuses_count', 
                          'actor_verified',
                          'class',
                          'manually_tweeting',
                          'followers_friends_ratio',
                          'favourites_friends_ratio',
                          'favourites_followers_ratio',
                          'favourites_status_ratio',
                          'actor_registration_from_now'
                          ]

  tweets_original_cols = ['actor_favorites_count', 
                          'actor_followers_count', 
                          'actor_friends_count', 
                          'actor_listed_count', 
                          'actor_statuses_count', 
                          'actor_verified',
                          'tweet_hashtags_count',
                          'tweet_mentions_count',
                          'tweet_urls_count',
                          'class'
                          ]

  manual_generators = ['Twitter Web Client', 
                     'Twitter for iPhone', 
                     'Twitter for Android', 
                     'Twitter for BlackBerry', 
                     'Twitter for Windows Phone', 
                     'Twitter for iPad', 
                     'Twitter for BlackBerry\xc2\xae', 
                     'Twitter for Mac', 
                     'Twitter for Android Tablets', 
                     'Twitter for Windows', 
                     'Twitter for Apple Watch', 
                     'Twitter for  Android']

  def requires(self):
    yield [S3ToLocalTask(s3_path=self.s3_folder + s3_file, local_path=self.local_folder + s3_file) for s3_file in S3Client().list(path=self.s3_folder)]
    yield RangeDailyBase(start=self.start, of='EnrichTrainerData')

  def output(self):
    return LocalTarget(self.model_path())

  def run(self):
    brands = self.concat_dataframes(self.local_folder + '*.brand')
    brands['class'] = 0
    person = self.concat_dataframes(self.local_folder + '*.person')
    person['class'] = 1

    tweets = pd.concat([brands, person])
    del tweets['score']
    del tweets['tweet_symbols_count']
    del tweets['tweet_trends_count']

    tweets = self.generate_calculated_features(tweets)

    cols = tweets[self.tweets_features_cols].columns
    cols = cols - ['class']
    print '### Train Columns:', cols

    k_fold = KFold(n=len(tweets), n_folds=6, indices=False, shuffle=True)
    b_scores, svc_scores = [], []

    for train_indices, test_indices in k_fold:
      train = np.asarray(tweets[train_indices][cols])
      train_y    = np.asarray(tweets[train_indices]['class'])

      test = np.asarray(tweets[test_indices][cols])
      test_y     = np.asarray(tweets[test_indices]['class'])

      clf = RandomForestClassifier(n_estimators=25)
      clf.fit(train, train_y)
      clf_probs = clf.predict_proba(test)
      print confusion_matrix(test_y, clf.predict(test))
      print 'score:' + str(clf.score(test, test_y))
        
    forest = RandomForestClassifier(n_estimators=25)
    forest.fit(tweets[cols], tweets['class'])

    score = forest.score(tweets[cols], tweets['class'])
    print 'final score: ', score

    if not os.path.exists(self.directory):
      os.makedirs(self.directory)

    joblib.dump(forest, self.model_path(), compress=9)

  def model_path(self):
    return self.directory + self.date.strftime('actor_classification_random_forest_%Y%m%d.pkl')
    
  def actor_registration_from_now(self, registration):
    r = parser.parse(registration)
    d = relativedelta.relativedelta(datetime.now(tzlocal()), r)
    return d.years * 12 + d.months

  def generate_calculated_features(self, tweets):
    tweets = tweets.dropna(subset=self.tweets_original_cols)

    int_cols = ['actor_followers_count', 'actor_friends_count', 'actor_favorites_count', 'actor_statuses_count', 'actor_listed_count', 'class']
    tweets[int_cols] = tweets[int_cols].astype(int)

    tweets['manually_tweeting'] = tweets['tweet_generator'].apply(lambda entry: 1 if entry in self.manual_generators else 0)
    tweets['actor_verified'] = tweets['actor_verified'].apply(lambda entry: 1 if entry else 0)
    tweets['followers_friends_ratio'] = tweets['actor_followers_count']/tweets['actor_friends_count']
    tweets['favourites_friends_ratio'] = tweets['actor_favorites_count']/tweets['actor_friends_count']
    tweets['favourites_followers_ratio'] = tweets['actor_favorites_count']/tweets['actor_followers_count']
    tweets['favourites_status_ratio'] = tweets['actor_favorites_count']/tweets['actor_statuses_count']

    tweets['actor_registration_from_now'] = tweets['actor_created_at'].apply(lambda registration: self.actor_registration_from_now(registration))

    tweets = tweets.replace([np.inf, -np.inf], 0).dropna(subset=self.tweets_features_cols)

    return tweets

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
  servers = luigi.Parameter(default='root@staging.feed.encorealert.com')
  key_file = luigi.Parameter(default='/Users/felipeclopes/.ec2/encore')
  
  model_local_directory = luigi.Parameter(default='data/actor_classification/models/')
  model_remote_directory = luigi.Parameter(default='/mnt/luigi/models/actor_classification/')

  def requires(self):
    return TrainRandomForestModel(self.date)

  def output(self):
    return [RemoteTarget(self.model_path(self.model_remote_directory), host=server, key_file=self.key_file) for server in self.servers]

  def run(self):
    for server in self.servers.split(","):
      LocalToRemoteTask(host=server, remote_path=self.model_path(self.model_remote_directory), key_file=self.key_file, 
        local_path=self.model_path(self.model_local_directory)).run()

  def model_path(self, directory):
    return directory + self.date.strftime('actor_classification_random_forest_%Y%m%d.pkl')

if __name__ == "__main__":
    luigi.run()