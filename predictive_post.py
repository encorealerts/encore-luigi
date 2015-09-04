import logging
import luigi
import redis

import mysql.connector

from datetime import datetime
from luigi.contrib.mysqldb import MySqlTarget

class AggregateToRedis(luigi.Task):

  db_host     = luigi.configuration.get_config().get('mysql', 'host', 'localhost')
  db_port     = luigi.configuration.get_config().get('mysql', 'port', 3306)
  db_database = luigi.configuration.get_config().get('mysql', 'database', 'encore_analytics_development')
  db_user     = luigi.configuration.get_config().get('mysql', 'user', 'root')
  db_password = luigi.configuration.get_config().get('mysql', 'password', '')

  redis_host  = luigi.configuration.get_config().get('redis', 'host', 'localhost')
  redis_port  = luigi.configuration.get_config().get('redis', 'port', 6379)
  redis_db    = luigi.configuration.get_config().get('redis', 'db', 0)

  date = luigi.Parameter(default=datetime.today())

  query =  "SELECT MAX(engagement_score) as max_engagement, \
              AVG(engagement_score) as avg_engagement, \
              STD(engagement_score) as std_engagement, \
              COUNT(engagement_score) as count, \
              rule_id as rule_id \
            FROM \
              predictive_post_data \
            WHERE \
              created_at > DATE(%(date)s) \
            GROUP BY \
              rule_id" 

  def output(self):
    return MySqlTarget(host=self.db_host, database=self.db_database, user=self.db_user, password=self.db_password, table='predictive_post', update_id=int(self.date.strftime("%Y%m%d%H")))

  def run(self):
    print ('ruunnnnnn!!')
    connection = self.connect()
    cursor = connection.cursor(buffered=True)

    cursor.execute(self.query, {'date': self.date})

    redis_client = self.redis_client()
    for (_max, avg, std, count, rule) in cursor:
      print '_max:', _max, 'avg', avg, 'std', std, 'count', count, 'rule', rule
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