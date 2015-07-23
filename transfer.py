# Copyright (c) 2014 Mortar Data
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import abc
import datetime

import luigi
from luigi import configuration, LocalTarget
from luigi.parameter import Parameter
from luigi.s3 import S3Target, S3PathTask, S3Client
from luigi.contrib.ssh import RemoteTarget

from ConfigParser import ConfigParser, NoOptionError, NoSectionError

class TransferTask(luigi.Task):
    """
    Superclass Luigi Task to move data between S3 and local file
    systems. Don't instantiate this directly, but rather use
    :py:class:`LocalToS3Task` or :py:class:`S3ToLocalTask`.
    """

    def output(self):
        """
        Task output. Returns the Target for the copy operation,
        so that the copy won't happen if the file already exists.

        :rtype: list of Target:
        :returns: list of Target for the copy operation
        """
        return [self.output_target()]

    @abc.abstractmethod
    def input_target(self):
        raise RuntimeError("Please implement input_target method")

    @abc.abstractmethod
    def output_target(self):
        raise RuntimeError("Please implement output_target method")

    def run(self):
        """
        Run the copy operation.
        """
        if not self.output_target().exists():
            try:
                r = self.input_target().open('r')
                w = self.output_target().open('w')
                w.write(r.read())
                w.close()
            except:
                basic_error = "Error in writing from %s to %s" % (self.input_target().path, self.output_target().path)
                if self.output_target().exists():
                    try:
                        self.output().remove()
                    except:
                        raise RuntimeError(basic_error + " and an error occured while deleting partially written file")
                raise
                raise RuntimeError(basic_error )
        else:
            raise RuntimeError("Target %s already exists!" % self.output_target().path)

class S3TransferTask(TransferTask):
    """
    Superclass Luigi Task to move data between S3 and local file
    systems. Don't instantiate this directly, but rather use
    :py:class:`LocalToS3Task` or :py:class:`S3ToLocalTask`.
    """

    @abc.abstractmethod
    def input_target(self):
        raise RuntimeError("Please implement input_target method")

    @abc.abstractmethod
    def output_target(self):
        raise RuntimeError("Please implement output_target method")

    def _get_s3_client(self):
        if not hasattr(self, "client"):
            self.client = \
                S3Client(
                    luigi.configuration.get_config().get('s3', 'aws_access_key_id'), 
                    luigi.configuration.get_config().get('s3', 'aws_secret_access_key'))
        return self.client

class LocalToRemoteTask(TransferTask):
    # full (absolute) local path to the file to copy
    remote_path = Parameter()
    host = Parameter()
    key_file = Parameter()

    # S3 URL for the file destination (including file name)
    local_path = Parameter()

    def input_target(self):
        return LocalTarget(self.local_path)

    def output_target(self):
        return RemoteTarget(self.remote_path, host=self.host, key_file=self.key_file)

class LocalToS3Task(S3TransferTask):
    """
    Copy a file from local disk to S3.

    To use this class, define the following section in your Luigi 
    client configuration file:

    ::[s3]
    ::aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    ::aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}

    Example usage:

    ::    # copy from /mnt/tmp/myfile.txt to s3://my-bucket/my-folder/myfile.txt
    ::    LocalToS3Task(local_path='/mnt/tmp/myfile.txt', 
    ::                  s3_path='s3://my-bucket/my-folder/myfile.txt')
    """

    # full (absolute) local path to the file to copy
    local_path = Parameter()

    # S3 URL for the file destination (including file name)
    s3_path = Parameter()

    def input_target(self):
        return LocalTarget(self.local_path)

    def output_target(self):
        return S3Target(self.s3_path, client=self._get_s3_client())

class RemoteToS3Task(S3TransferTask):
    host = Parameter()
    # full (absolute) local path to the file to copy
    remote_path = Parameter()
    # S3 URL for the file destination (including file name)
    s3_path = Parameter()
    # Keyfile path
    key_file = Parameter()

    def input_target(self):
        return RemoteTarget(self.remote_path, host=self.host, key_file=self.key_file)

    def output_target(self):
      return S3Target(self.s3_path, client=self._get_s3_client())

class S3ToLocalTask(S3TransferTask):
    # S3 URL for the file to copy
    s3_path = Parameter()

    # full (absolute) target local path for the file (including file name)
    local_path = Parameter()

    def input_target(self):
        return S3Target(self.s3_path, client=self._get_s3_client())

    def output_target(self):
        return LocalTarget(self.local_path)