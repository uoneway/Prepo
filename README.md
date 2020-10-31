# plink
웹페이지 scraping부터 이를 topic modeling까지 쉽게 하기 위한 프로젝트입니다.   
Support 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) 

본 프로젝트의 기반이 된 라이브러리는 다음과 같으며 submodule로 포함되어 있습니다.
- Top2Vec
- newspaper


## TopicModel 
Topic modeling을 위한 class입니다.
다음의 알고리즘을 사용하게 됩ㄴ디ㅏ.
1. Document/word enbedding: Doc2Vec DBOW 를 통해 docs와 word를 동일한 벡터공간에 표현
2. Dimension reduction: UMAP
3. Clustering: HDBSCAN
    - tree 하나 전체에 대해 일괄적인 동일 기준으로 잘라서 clustering을 구하는 HDBSCAN과 달리,
    줄기별로(서로 density가 다른) 별도의 기준으로 자르는 방법
        - tree를 자르는 기준이 그 줄기 안에서 왔다갔다 했을 때 cluster가 분화되는지 아닌지 안정성을 기준으로 자르게 됨
        - 하지만 각각 장단점이 있음. HDBSCAN은 큰 클러스터를 너무 세부적으로 자르는 점이 있어, Epsilon 값을 조정하면 큰 클러스터에 대해서는 DBSCAN을, 작은 클러스터에 대해서는 HDBSCAN 을 함께 사용할 수 있음
4. Find topic vector
5. Find n-closest word vectors(keyword) to the resulting topic vector

### TopicModel class 사용법
#### TopicModel 생성하기

```python
tm_model = TopicModel(user_docs_df['contents_prep'], doc_ids=user_docs_df.index)
```

#### topic 정보 가져오기

```python
topics_info = tm_model.get_topics_info()
topics_info 
#[{'topic_idx': 0, 'topic_words': array(['neuroscience', 'neuroimaging', ...,])},
# [{'topic_idx': 1, ...
```

#### topic reduction

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

#### topic 내 document id 가져오기

```python
topic_idx_user_input = 1
doc_ids = tm_model.get_doc_ids_by_topic(topic_idx_user_input)
print(doc_ids)
```