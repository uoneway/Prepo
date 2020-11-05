import sys
import os
import unittest
import time
import traceback
import re
from collections import defaultdict, OrderedDict
import concurrent.futures

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(PARENT_DIR)

# newspaper's unit tests are in their own separate module, so
# insert the parent directory manually to gain scope of the
# core module
sys.path.insert(0, PARENT_DIR)

DATA_PATH = os.path.join(TEST_DIR, 'data', 'text')

#import prepo
from prepo import utils  # pylint: disable=import-error
from prepo.scraper import scrap  # pylint: disable=import-error
from prepo.preprocessor import preprocessing, summarize  # pylint: disable=import-error
from prepo.topic_model import TopicModel  # pylint: disable=import-error
#from submodules.Top2Vec.top2vec import Top2Vec


test_urls =["onedrive.live.com/sdfsdf",
        "https://www.nytimes.com/2020/10/20/us/politics/stimulus-deal-mitch-mcconnell-nancy-pelosi.html?action=click&module=Top%20Stories&pgtype=Homepage",
        "https://news.naver.com/main/ranking/read.nhn?mid=etc&sid1=111&rankingType=popular_day&oid=015&aid=0004435281&date=20201021&type=1&rankingSeq=7&rankingSectionId=101",
        "https://planbs.tistory.com/entry/Git-Pull%EC%97%90%EC%84%9C-%EC%B6%A9%EB%8F%8C-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0",
        "planbs.tistory.com/entry/Git-Pull%EC%97%90%EC%84%9C-%EC%B6%A9%EB%8F%8C-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0",
        "www.nytimes.com/2020/10/20/us/politics/stimulus-deal-mitch-mcconnell-nancy-pelosi.html?action=click&module=Top%20Stories&pgtype=Homepage",]

docs_info, docs_idx, no_scraped_urls_by_types = scrap(test_urls, sensitive_domain_cats=['ott', 'cloud'])
print(no_scraped_urls_by_types)

docs =[]
docs_p =[]
for doc_info in docs_info:
        doc = doc_info['contents']
        docs.append(doc)
        docs_p.append(preprocessing(doc))

for i in zip(docs,docs_p):
        print(i)






