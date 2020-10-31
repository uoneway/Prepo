import sys
import os
from pathlib import Path
import numpy as np 
import pandas as pd
import logging

PROJECT_DIR = os.path.abspath(os.path.dirname("__file__")) #"/home/lab13/RunToLearn/plink/"
# sys.path.insert(0, PROJECT_DIR + '/src')
# sys.path.insert(1, PROJECT_DIR + '/submodules/Top2Vec')
# sys.path.insert(1, PROJECT_DIR + '/submodules')
#print(sys.path)

from src import utils
from src.scrap import scraper
from src.preprocess import preprocessing, summarizer
from submodules.Top2Vec.top2vec import Top2Vec
from models.topic_model import TopicModel
# import utils
# from scrap import scraper
# from preprocess import preprocessing, summarizer
# from top2vec import Top2Vec



DATA_DIR = PROJECT_DIR + '/datasets/'
MODEL_DIR = PROJECT_DIR + '/models/'
IMAGES_DIR = PROJECT_DIR + '/images/'
TEST_DIR = PROJECT_DIR + '/test/'

## logger
logger = logging.getLogger()
logging.basicConfig(level=logging.debug)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 콘솔로 출력
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

###############################################################################

## 데이터 가져오기/스크랩하기
user_data_dir = TEST_DIR + "data/choi_urls/"
user_docs_info_data_filename = 'docs_info_df.pkl'
user_urls_data_filename = 'choi_time_url_df.pkl'

DO_SCRAP = False
if Path(user_data_dir + user_docs_info_data_filename).exists() and not DO_SCRAP:
    docs_info_df = utils.load_obj(user_data_dir, user_docs_info_data_filename)

else:
    input_df = utils.load_obj(user_data_dir, user_urls_data_filename)

    docs_info, docs_idx, error_urls_by_types = scraper(input_df['url'], input_df.index)  # .tolist()
    docs_info_df = pd.DataFrame.from_dict(docs_info)
    docs_info_df.index = docs_idx

    docs_info_df = docs_info_df.join(input_df['time'], how='left')
    docs_info_df.rename(columns = {"time": "clip_at"}, inplace=True)
    docs_info_df = docs_info_df.sort_values(by=['clip_at'], axis=0).reset_index(drop=True)  # 정렬 후 reset index

    # print(docs_info_df)
    # print(docs_info_df.isnull().sum())
    # print(error_urls_by_types)

    # utils.save_obj(user_data_dir, user_docs_info_data_filename, docs_info_df)
    # utils.save_obj(user_data_dir, "error_urls_by_types.txt", error_urls_by_types)


#logger.info(docs_info_df)
#print(docs_info_df)


## 전처리 적용
DO_PREP = False
user_docs_info_prep_data_filename = 'docs_info_prep_df.pkl'
if Path(user_data_dir + user_docs_info_prep_data_filename).exists() and not DO_PREP:
    docs_info_prep_df = utils.load_obj(user_data_dir, user_docs_info_prep_data_filename)

else:
    # 제목과 contents 부분을 전처리 후 붙여주기
    docs_info_prep_df = docs_info_df.copy()
    docs_info_prep_df['contents_prep'] = docs_info_prep_df['title'] + ". " + docs_info_prep_df['contents']
    docs_info_prep_df['contents_prep'] = docs_info_prep_df['contents_prep'].apply(preprocessing)
    docs_info_prep_df['contents_prep'] = docs_info_prep_df['contents_prep'].apply(summarizer)
    utils.save_obj(user_data_dir, 'docs_info_prep_df.pkl', docs_info_prep_df)


#### 추후 삭제
user_docs_df =  docs_info_prep_df.iloc[:-60, :]
user_docs_post1_df = docs_info_prep_df.iloc[-60:-30, :]
user_docs_post2_df = docs_info_prep_df.iloc[-30:, :]


tm_model = TopicModel(user_docs_df['contents_prep'], doc_ids=user_docs_df.index)

## topic 정보 가져오기
# [{'topic_idx': 0, 'topic_words': array(['neuroscience', 'neuroimaging', ...,])},
# [{'topic_idx': 1, ...
topics_info = tm_model.get_topics_info()

print(tm_model.get_topic_sizes())
## topic reduction
topic_num_user_input = 10
tm_model.reduce_topic(topic_num_user_input)
topics_info = tm_model.get_topics_info()
print(topics_info)

# document 추가/제거하기
# tm_model.add_documents(documents, doc_ids=doc_ids)
# tm_model.delete_documents(doc_ids)

print(tm_model.get_topic_sizes(reduced=True))
# topic 내 document id 가져오기
topic_idx_user_input = 1
doc_ids = tm_model.get_doc_ids_by_topic(topic_idx_user_input)
print(doc_ids)

