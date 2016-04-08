import os
import fnmatch

import pandas as pd

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

import glob
import luigi

from boto.s3.connection import S3Connection

from luigi import LocalTarget
from luigi.s3 import S3Target, S3Client
from luigi.parameter import Parameter
from luigi.tools.range import RangeDailyBase
from luigi.contrib.ssh import RemoteTarget

from transfer import RemoteToS3Task, S3ToLocalTask, LocalToS3Task, LocalToRemoteTask

from meltwater_smart_alerts.ml.pipeline import *

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
                      ("debugger1", PipelineDebugger('Starting')),
                      ("qt_words", NumberOfWords(["name","summary"])),
                      ("qt_non_alphanum_chars", NumberNonAlphaNumChars(["name","summary"])),
                      ("qt_upper_case_chars", NumberUpperCaseChars(["name","summary"])),
                      ("qt_camel_case_words", NumberCamelCaseWords(["name","summary"])),
                      ("qt_mentions", NumberOfMentions(["summary"])),
                      ("qt_periods", NumberOfPeriods(["summary"])),
                      ("avg_words_per_period", AvgWordsPerPeriod(["summary"])),
                      ("lower_case", TextToLowerCase(["screen_name","name","summary"])),
                      ("family", MentionToFamilyRelation(["summary"])),
                      ("debugger2", PipelineDebugger('Basic Statistics')),
                      ("person_names", PersonNames(["name"])),
                      ("debugger3", PipelineDebugger('Person Names')),
                      ("occupations", MentionToOccupation(["summary"])),
                      ("debugger4", PipelineDebugger('Occupations')),
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
                      ("debugger5", PipelineDebugger('Names TFIDF')),
                      ("screen_name_tfidf", DataFrameTfidfVectorizer(col="screen_name", 
                                            ngram_range=(3, 5), 
                                            analyzer="char",
                                            binary=True, #False
                                            min_df = 50,
                                            max_features=50)),
                      ("debugger6", PipelineDebugger('Screen Names TFIDF')),
                      ("summary_tfidf", DataFrameTfidfVectorizer(col="summary",
                                          token_pattern=r'\w+',
                                          ngram_range=(1, 3), 
                                          analyzer="word",
                                          binary=True, #False
                                          sublinear_tf=True, 
                                          stop_words='english',
                                          min_df = 50,
                                          max_features=50)),
                      ("debugger7", PipelineDebugger('Summary TFIDF')),
                      ("drop_text_cols", DropColumnsTransformer(["screen_name","name","summary"])),
                      ("nparray", NumpyArrayTransformer()),
                      ("debugger8", PipelineDebugger('Finish')),
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
