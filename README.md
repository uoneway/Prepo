# Prepo(Personalized Report)
웹콘텐츠 **scraping**부터 **preprocessing**, **topic modeling**, **summarization** 까지 일련의 과정을 구현한 라이브러리입니다.
**개인이 학습하고자 하는 웹페이지 콘텐츠를 자동으로 수집/분류/요약하여 리포트를 생성**해주는 [Prepo](https://github.com/yerachoi/prepo-flask-gentelella) 서비스의 기반이 되는 라이브러리입니다.

[![ReadMe Card](https://github-readme-stats.vercel.app/api/pin/?username=yerachoi&repo=prepo-flask-gentelella)](https://github.com/yerachoi/prepo-flask-gentelella)
[![ReadMe Card](https://github-readme-stats.vercel.app/api/pin/?username=uoneway&repo=Top2Vec&show_owner=True)](https://github.com/uoneway/Top2Vec)

기능별로 다음의 라이브러리에 기반하고 있습니다. 
- Scrap: [Newspaper3k](https://github.com/codelucas/newspaper)
- Preprocessing
  - MeCab
  - Sentencepiece
- Summarization: gensim.summarization.summarizer(TextRank)
- Topic modeling: [Top2Vec](https://github.com/uoneway/Top2Vec)


## Top2Vec
Top2Vec은 topic modeling을 위한 모델로 다양한 알고리즘의 집합이라 할 수 있습니다.
다음 단계를 통해 topic modeling을 수행합니다. 

1. Document/word enbedding
    기본적으로는 Doc2Vec(DBOW)를 사용하고 있지만, 다음의 Pre-trained model도 쉽게 사용 가능합니다.
    - Universal Sentence Encoders
    - BERT Sentence Transformer
2. Dimension reduction: UMAP
3. Clustering: HDBSCAN
    - tree 하나 전체에 대해 일괄적인 동일 기준으로 잘라서 clustering을 구하는 HDBSCAN과 달리,
    줄기별로(서로 density가 다른) 별도의 기준으로 자르는 방법
        - tree를 자르는 기준이 그 줄기 안에서 왔다갔다 했을 때 cluster가 분화되는지 아닌지 안정성을 기준으로 자르게 됨
        - 하지만 각각 장단점이 있음. HDBSCAN은 큰 클러스터를 너무 세부적으로 자르는 점이 있어, Epsilon 값을 조정하면 큰 클러스터에 대해서는 DBSCAN을, 작은 클러스터에 대해서는 HDBSCAN 을 함께 사용할 수 있음
4. Find topic vector
5. Find n-closest word vectors(keyword) to the resulting topic vector

### TopicModel class 기본 사용법

Top2Vec 라이브러리는 상당히 잘 구성되어 있으나 사용하기가 불편한 단점이 있어 이를 상속한TopicModel class를 통해 이를 편하게 사용할 수 있도록 재구성하였습니다. 

#### TopicModel 생성하기

```python
tm_model = TopicModel(user_docs_df['contents_prep'], doc_ids=user_docs_df.index)
```

#### topic num reduction
HDBSCAN으로 도출된 cluster는 hierachy를 가지고 있습니다. 따라서 다시 topic modeling을 하지 않고도 사용자가 원하는 topic 수대로 기존 도출된 topic들을 유사한 것 끼리 다시 묶어줄 수 있습니다.
이렇게 원하는 topic 수로 토픽을 다루고 싶다면, 다음 함수를 무조건 한 번은 호출해줘야 합니다.

```python
topic_num_user_input = 10
tm_model.reduce_topic(topic_num_user_input)
topics_info = tm_model.get_topics_info()
print(topics_info)
```

#### document 추가/제거하기

```python
tm_model.add_documents(documents, doc_ids=doc_ids)
tm_model.delete_documents(doc_ids)
```

### topic, docs, keyword, 벡터 정보 가져오기

#### topic 정보 가져오기

```python
# 전체가져오기
topics_info = tm_model.get_topics_info()
#[{'topic_idx': 0, 'topic_words': array(['neuroscience', 'neuroimaging', ...,])},
# [{'topic_idx': 1, ...

# 특정 토픽의 정보만 가져오기
get_topic_info(self, topic_idx, is_reduced=None)

# docs -> topics
get_documents_topics(self, doc_ids, reduced=False)

# keywords -> topic
search_topics(self, keywords, num_topics, keywords_neg=None, reduced=False):
```

#### doc 정보 가져오기

```python
# topic -> docs
get_doc_ids_by_topic(topic_idx, is_reduced=None)
return doc_ids

# keywords -> doc
search_documents_by_keywords(self, keywords, num_docs, keywords_neg=None, return_documents=True):

# docs -> docs
get_docs_by_doc(self, doc_ids, num_docs=3):
return doc_ids
```

#### keyword정보 가져오기

```python
# topic -> keyword
get_topics_info():
				# {'topic_idxes': [topic1_idx, topic2_idx,..]
        # 'topics_words': [topic_words of topic1,]   word도 리스트
        # 'topic_vectors': [topic_vector,]}  vector도 리스트
    return self.topics_info

# keywords -> keywords
similar_words(keywords, num_words, keywords_neg=None):

# docs -> keyword
get_keywords_by_doc(doc_ids, doc_ids_neg=None,)
return words, word_scores

# docs ->hot keyword
get_hot_keywords_by_docs(self, doc_ids, doc_weights, doc_ids_neg=None, num_words=10)
	return np.array(hot_words), np.array(hot_word_scores)
```

#### 벡터 정보 가져오기

```python
get_2d_vectors(is_reduced=None):
	return topics_idx_vector, docs_idx_vector, words_idx_vector
```


