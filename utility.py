
import os
import errno
import logging
from datetime import datetime, timedelta
import re
import shutil


class StreamToLogger(object):
    """
    Redirect all the stdout/err to the logger, therefore both print and traceback
    are redirected to logger
    """

    def __init__(self, logger, LogFile='log', log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.logFile = LogFile

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=self.logFile,
            filemode='a'
        )
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# utility function
def makedir(path):  # se esiste già non fa nulla e salta l'exceprtion
    """
    Make dir only is it doesn't exist yet
    :param path: path to the folder that is to be created
    :return:
    """
    try:
        os.makedirs(path)
        print("make " + path + " dir")
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        pass
    return


def deleteContentFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
def GetTime(s):
    '''

    :param s: seconds (int)
    :return: the days hours minutes and second thet correspond to the seconds input, as string
    '''
    sec = timedelta(seconds=s)
    #sec = timedelta(seconds=int(input('Enter the number of seconds: ')))
    d = datetime(1, 1, 1) + sec

    #print("DAYS:HOURS:MIN:SEC")
    #print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))
    t = "%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second)
    return t

def logcleaner(pathFile):
    """
    Clean a text file
    :param pathFile:
    :return:
    """
    string = open(pathFile).read()
    new_str = re.sub('[^a-zA-Z0-9\n\.\-=<>~:,\[\]\t_(){}/]', ' ', string)
    with open(pathFile, 'w') as logfile:
        logfile.write(new_str)

    return