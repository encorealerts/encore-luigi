import logging
import luigi
import redis

import mysql.connector

from datetime import datetime
from datetime import timedelta
from luigi.contrib.mysqldb import MySqlTarget

# Search for PredictivePostConfigurations and pass rule_id and segment as
# parameter for the AggregateToRedis task
class ConfigsToAggregation(luigi.Task):

  db_host     = luigi.configuration.get_config().get('mysql', 'host', 'localhost')
  db_port     = luigi.configuration.get_config().get('mysql', 'port', 3306)
  db_database = luigi.configuration.get_config().get('mysql', 'database', 'encore_analytics_development')
  db_user     = luigi.configuration.get_config().get('mysql', 'user', 'root')
  db_password = luigi.configuration.get_config().get('mysql', 'password', '')

  query =  "SELECT rule_id as rule, \
              segment as segment \
            FROM \
              predictive_post_configurations" 

  def requires(self):
    connection = self.connect()
    cursor = connection.cursor(buffered=True)
    cursor.execute(self.query)

    date = datetime.today() - timedelta(days=5)

    return map(lambda (rule, segment): AggregateToRedis(date=date, rule=rule, segment=segment), 
               cursor)

  def connect(self):
    connection = mysql.connector.connect(user=self.db_user,
                                         password=self.db_password,
                                         host=self.db_host,
                                         port=self.db_port,
                                         database=self.db_database)
    return connection

class AggregateToRedis(luigi.Task):

  db_host     = luigi.configuration.get_config().get('mysql', 'host', 'localhost')
  db_port     = luigi.configuration.get_config().get('mysql', 'port', 3306)
  db_database = luigi.configuration.get_config().get('mysql', 'database', 'encore_analytics_development')
  db_user     = luigi.configuration.get_config().get('mysql', 'user', 'root')
  db_password = luigi.configuration.get_config().get('mysql', 'password', '')

  redis_host  = luigi.configuration.get_config().get('redis', 'host', 'localhost')
  redis_port  = luigi.configuration.get_config().get('redis', 'port', 6379)
  redis_db    = luigi.configuration.get_config().get('redis', 'db', 0)

  date    = luigi.DateParameter(default=datetime.today() - timedelta(days=5))
  rule    = luigi.Parameter()
  segment = luigi.Parameter()

  query =  "SELECT MAX(p.engagement_score) as max_engagement, \
              AVG(p.engagement_score) as avg_engagement, \
              STD(p.engagement_score) as std_engagement, \
              COUNT(p.engagement_score) as count, \
              p.rule_id as rule_id \
            FROM \
              predictive_post_data p \
            JOIN \
              activities at ON p.native_id = at.native_id \
            JOIN \
              actors ac ON at.actor_id = ac.id  \
            WHERE \
              p.created_at > DATE('%(date)s') \
            AND \
              p.rule_id = %(rule)d \
              %(segment_clause)s \
            GROUP BY \
              rule_id "

  def input(self):
    return LocalTarget(self.input_file())

  def output(self):
    update_id = "{0}-{1}".format(self.date.strftime("%Y%m%d%H"), self.rule)
    return MySqlTarget(host=self.db_host, 
                       database=self.db_database, 
                       user=self.db_user, 
                       password=self.db_password, 
                       table='predictive_post', 
                       update_id=update_id)

  def run(self):
    print 'AggregateToRedis - rule:', self.rule, 'date:', self.date, 'segment:', self.segment
    
    connection = self.connect()
    cursor = connection.cursor(buffered=True)

    segment_clause = "" if self.segment == None else "AND (ac.segment = '{0}' OR ac.segment IS NULL)".format(self.segment)

    query = self.query % {  'date': self.date, 
                            'rule': self.rule, 
                            'segment_clause': segment_clause}

    cursor.execute(query)

    redis_client = self.redis_client()
    for (_max, avg, std, count, rule) in cursor:
      print 'AggregateToRedis - _max:', _max, 'avg:', avg, 'std:', std, 'count:', count, 'rule:', rule
      if count < 30:
        print 'AggregateToRedis - skipping - rule:', rule
        continue
      redis_client.hset("encore:predictive-post-%s" % rule, 'max', _max)
      redis_client.hset("encore:predictive-post-%s" % rule, 'avg', avg)
      redis_client.hset("encore:predictive-post-%s" % rule, 'std', std)
      redis_client.hset("encore:predictive-post-%s" % rule, 'count', count)

    self.output().touch()

  def connect(self):
    connection = mysql.connector.connect(user=self.db_user,
                                         password=self.db_password,
                                         host=self.db_host,
                                         port=self.db_port,
                                         database=self.db_database)
    return connection

  def redis_client(self):
    return redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db)

if __name__ == "__main__":
    luigi.run()
