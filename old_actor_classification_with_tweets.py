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

import nltk
from nltk import word_tokenize

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer

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
  
  input_prefix = 'data/actor_classification/raw/twitter-actor.csv'
  output_prefix = 'data/actor_classification/csv/enriched-twitter-actor.csv'

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

    print self.input_file()
    df = self.load_dataframe(self.input_file())

    self.output()[0].makedirs()
    business_data = self.lookup_twitter_statuses(df[df['class'] == 'business'])
    self.persist_complete_data(self.output_file() + '.brand', business_data)
    self.output()[1].makedirs()
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
        try:
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
        except Exception, ex:
          print ex


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
      print 'Downloaded tweet info for', len(tweets)
      time.sleep(5)

    return tweets

class TrainRandomForestModel(luigi.Task):
  date         = luigi.Parameter(default=datetime.today())
  start        = luigi.Parameter(default=datetime(2015,07,01))
  
  s3_csvs      = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/csv/')
  s3_models    = luigi.Parameter('s3://encorealert-luigi-development/actor_classification/models/')
  
  local_csvs   = '/mnt/encore-luigi/data/actor_classification/csv/'
  local_path  = '/mnt/encore-luigi/data/actor_classification/models/'

  tweets_features_cols = ['actor_summary',
                          'actor_favorites_count', 
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

  tweets_original_cols = ['actor_summary',
                          'actor_favorites_count', 
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
    yield [S3ToLocalTask(s3_path=self.s3_csvs + s3_file, local_path=self.local_csvs + s3_file) for s3_file in S3Client().list(path=self.s3_csvs)]
    yield RangeDailyBase(start=self.start, of='EnrichTrainerData')

  def output(self):
    return {
      'model': S3Target(self.model_path(self.s3_models)),
      'counter': S3Target(self.counter_path(self.s3_models))
      }

  def run(self):
    brands = self.concat_dataframes(self.local_csvs + '*.brand')
    brands['class'] = 0
    person = self.concat_dataframes(self.local_csvs + '*.person')
    person['class'] = 1

    tweets = pd.concat([brands, person])
    del tweets['score']
    del tweets['tweet_symbols_count']
    del tweets['tweet_trends_count']

    tweets = self.generate_calculated_features(tweets)
    tweets, counter = self.create_ngrams_with_bio(tweets)

    cols = tweets.columns
    cols = cols - ['class']
    print '### Train Columns:', cols

    for fold in [6,2]:
      k_fold = KFold(n=len(tweets), n_folds=fold, indices=False, shuffle=True)
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

    if not os.path.exists(self.local_path):
      os.makedirs(self.local_path)

    joblib.dump(forest, self.model_path(self.local_path), compress=9)
    joblib.dump(counter, self.counter_path(self.local_path), compress=9)

    with open(self.model_path(self.local_path)) as model_pickle:
      with open(self.counter_path(self.local_path)) as counter_pickle:
        with self.output()['model'].open(mode='w') as s3_model:
          with self.output()['counter'].open(mode='w') as s3_counter:
            s3_model.write(model_pickle.read())
            s3_counter.write(counter_pickle.read())


    os.remove(self.model_path(self.local_path))
    os.remove(self.counter_path(self.local_path))

  def model_path(self, directory):
    return directory + self.date.strftime('actor_classification_random_forest_%Y%m%d.pkl')
  def counter_path(self, directory):
    return directory + self.date.strftime('bio_count_vectorizer_%Y%m%d.pkl')
    
  def actor_registration_from_now(self, registration):
    r = parser.parse(registration)
    d = relativedelta.relativedelta(datetime.now(tzlocal()), r)
    return d.years * 12 + d.months

  def generate_calculated_features(self, tweets):
    tweets = tweets.dropna(subset=self.tweets_original_cols)

    int_cols = ['actor_followers_count', 'actor_friends_count', 'actor_favorites_count', 'actor_statuses_count', 'actor_listed_count', 'class']
    tweets[int_cols] = tweets[int_cols].astype(int)

    tweets.loc[:, ('manually_tweeting')] = tweets.loc[:, 'tweet_generator'].apply(lambda entry: 1 if entry in self.manual_generators else 0)
    tweets.loc[:, ('actor_verified')] = tweets.loc[:, 'actor_verified'].apply(lambda entry: 1 if entry else 0)
    tweets.loc[:, ('followers_friends_ratio')] = tweets.loc[:, 'actor_followers_count']/tweets.loc[:, 'actor_friends_count']
    tweets.loc[:, ('favourites_friends_ratio')] = tweets.loc[:, 'actor_favorites_count']/tweets.loc[:, 'actor_friends_count']
    tweets.loc[:, ('favourites_followers_ratio')] = tweets.loc[:, 'actor_favorites_count']/tweets.loc[:, 'actor_followers_count']
    tweets.loc[:, ('favourites_status_ratio')] = tweets.loc[:, 'actor_favorites_count']/tweets.loc[:, 'actor_statuses_count']

    tweets.loc[:, ('actor_registration_from_now')] = tweets.loc[:, 'actor_created_at'].apply(lambda registration: self.actor_registration_from_now(registration))

    tweets = tweets.loc[:, (self.tweets_features_cols)]
    tweets = tweets.replace([np.inf, -np.inf], 0).dropna()

    return tweets

  import nltk
  from nltk import word_tokenize
  from sklearn.feature_extraction.text import CountVectorizer

  # #######
  def tokenize(self, text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [('NUM' if word.isdigit() else word) for word in tokens]
    return tokens
  # ######## 

  def create_ngrams_with_bio(self, tweets):
    bio_words_countvect = CountVectorizer(tokenizer=self.tokenize,
                                                ngram_range=(1, 3), 
                                                analyzer="word",
                                                min_df = 30)

    bio_word_matrix = bio_words_countvect.fit_transform(tweets['actor_summary'])
    bio_word_matrix_df = pd.DataFrame(bio_word_matrix.A, columns=map(lambda name: 'bio_words_' + name, bio_words_countvect.get_feature_names()))
    
    del tweets['actor_summary']

    # vocabulary = joblib.load('<my vectorizer file>') 
    # vect = CountVectorizer(tokenizer=self.tokenize, ngram_range=(1,3), vocabulary = vocabulary)
    return tweets.join(bio_word_matrix_df), bio_words_countvect.get_feature_names()


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
  
  model_local_directory = 'data/actor_classification/deploy/'

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