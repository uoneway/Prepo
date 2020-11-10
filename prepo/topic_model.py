from Top2Vec.top2vec import Top2Vec
import numpy as np
import umap

class TopicModel(Top2Vec):
    """
    """

    def __init__(self,
                 documents, 
                 doc_ids,                 
                 umap_args=None, 
                 hdbscan_args=None,                 
                 ):
        ## docs
        self.documents = documents
        self.doc_ids = doc_ids

        ## model hyper-parameters
        # embedding_model은 tf hub를 사용하여 한 번 받으면 캐싱되어 그 이후에는 재다운 받지 않음
        # universal-sentence-encoder-multilingual
        # https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
        # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) 
        
        self.umap_args = {'n_neighbors': 5,
                        'n_components': 5,
                        'metric': 'cosine'
                        } if umap_args is None else umap_args
        self.hdbscan_args = {'min_samples': 2,
                            'min_cluster_size': 3,
                            #'cluster_selection_epsilon': 0.5,
                            # 'cluster_selection_method':'leaf'
                            } if hdbscan_args is None else hdbscan_args

        self.is_reduced = False

        super().__init__(self.documents,
                        document_ids = self.doc_ids,
                        embedding_model='universal-sentence-encoder-multilingual',  # distiluse-base-multilingual-cased
                        min_count=10,
                        workers=16,   # deep-learn'
                        keep_documents=True,
                        verbose=True,
                        umap_args=self.umap_args, 
                        hdbscan_args=self.hdbscan_args,
                        ) 

        # {'topic_idxes': [topic1_idx, topic2_idx,..]
        # 'topics_words': [topic_words of topic1,]   word도 리스트
        # 'topic_vectors': [topic_vector,]}  vector도 리스트
        self.topics_info = {}
        self.topics_info_reduced = {}

        self._update_topics_info()

        
    def get_topics_info(self, is_reduced=None):
        if is_reduced is None:
            is_reduced = self.is_reduced

        return self.topics_info_reduced if is_reduced \
            else self.topics_info

    def get_topics_info_as_dict_list(self, is_reduced=None):
        # [{'topic_idx': idx,
        # 'topic_words': [word1, word2 ...]
        # 'topic_vector': [vector1,vector2 ...]} ] 

        topics_info = self.get_topics_info(is_reduced)

        #list dict to dict list
        key_names = ['topic_idx', 'topic_words', 'topic_vector']
        topics_info_dict_list = [dict(zip(key_names, t)) for t in zip(*topics_info.values())]
        
        return topics_info_dict_list

    def get_topic_info(self, topic_idx, is_reduced=None):
        #{'topic_idx': idx,
        # 'topic_words': [word1, word2 ...]
        # 'topic_vector': [vector1,vector2 ...]}
        
        if is_reduced is None:
            is_reduced = self.is_reduced
        #super()._validate_topic_num(topic_idx, is_reduced)

        topics_info_dict_list = self.get_topics_info_as_dict_list(is_reduced)
        topics_info_dict = topics_info_dict_list[topic_idx]

        assert topics_info_dict_list[topic_idx]['topic_idx'] == topic_idx

        return topics_info_dict
        
    def _update_topics_info(self):

        # for not reduced
        topics_words, _, topic_idxes = self.get_topics()
        self.topics_info = {'topic_idxes': topic_idxes,
                            'topics_words': topics_words,
                            'topic_vectors': self.topic_vectors}
        # for reduced:
        if self.is_reduced:
            topics_words_reduced, _, topic_idxes_reduced = self.get_topics(reduced=True)       
            self.topics_info_reduced = {'topic_idxes': topic_idxes_reduced,
                                'topics_words': topics_words_reduced,
                                'topic_vectors': self.topic_vectors_reduced}

    def reduce_topic(self, num_topics):
        # 줄이고 싶다면 이 함수를 무조건 호출해줘야함 
        clusters_reduced_indice_list = self.hierarchical_topic_reduction(num_topics)
        self.is_reduced = True

        self._update_topics_info()


    def get_doc_ids_by_topic(self, topic_idx, is_reduced=None):
        if is_reduced is None:
            is_reduced = self.is_reduced
        if is_reduced:
            num_docs = super().get_topic_sizes(reduced=True)[0][topic_idx]
            docs, doc_scores, doc_ids = super().search_documents_by_topic(topic_idx, num_docs, reduced=True)
        else:
            num_docs = super().get_topic_sizes()[0][topic_idx]
            docs, doc_scores, doc_ids = super().search_documents_by_topic(topic_idx, num_docs)
        return doc_ids

    # def search_documents_by_documents(self, doc_ids, num_docs, doc_ids_neg=None, return_documents=True):
    #  If negative document ids are provided, the documents will be semantically dissimilar to those document ids.
    def get_docs_by_doc(self, doc_ids, num_docs=3):
        doc_scores, doc_ids = super().search_documents_by_documents(doc_ids, num_docs, return_documents=False)
        return doc_ids


    def get_keywords_by_docs(self, doc_ids, doc_ids_neg=None,):
        #리턴이 리스트의 리스트임!
    
        # (self, doc_ids, num_words, doc_ids_neg=None,):
        if doc_ids_neg is None:
            doc_ids_neg = []

        self._validate_doc_ids(doc_ids, doc_ids_neg)

        doc_indexes = self._get_document_indexes(doc_ids)
        # doc_indexes_neg = self._get_document_indexes(doc_ids_neg)

        doc_vecs = [self.document_vectors[ind] for ind in doc_indexes]
        # doc_vecs_neg = [self.document_vectors[ind] for ind in doc_indexes_neg]
        # combined_vector = self._get_combined_vec(doc_vecs, doc_vecs_neg)

        # 이름은 topic word 찾는거지만, 벡터받아서 가까운 50개 단어 추출해줌
        # combined_vector = np.expand_dims(combined_vector, axis=0)  # 1d라서 2d로 바꿔주기
        docs_words, docs_word_scores = super()._find_topic_words_and_scores(doc_vecs)

        return docs_words, docs_word_scores

    def get_hot_keywords_by_docs(self, doc_ids, doc_weights, doc_ids_neg=None, num_words=10):
        words, word_scores = self.get_keywords_by_doc(doc_ids, doc_ids_neg=doc_ids_neg,)

        # make weights of words
        # word_scores shape: doc_num * 50
        # doc_weights shape: doc_num
        doc_weights = np.array(doc_weights)
        doc_weights = doc_weights.reshape(-1,1)
        weights = doc_weights*word_scores

        # flatten
        words = np.ravel(words)
        word_scores = np.ravel(weights)

        # sort descending and return index
        index_order = np.flip(np.argsort(word_scores))[:num_words]
        hot_words = [words[index] for index in index_order]
        hot_word_scores = [word_scores[index] for index in index_order]

        return np.array(hot_words), np.array(hot_word_scores)

    def get_2d_vectors(self, is_reduced=None):
        if is_reduced is None:
            is_reduced = self.is_reduced
        # args for UMAP. Same as current args except n_components to plot graph
        umap_args_for_plot = self.umap_args.copy()
        umap_args_for_plot.update({'n_components': 2,})

        # Dimension reduction
        umap_model = umap.UMAP(**umap_args_for_plot, random_state=42).fit(super()._get_document_vectors())  #  + self.topic_words
        document_vectors_2d = umap_model.embedding_  # same as umap_model.transform(self._get_document_vectors())
        
        topics_info = self.get_topics_info(is_reduced)

        #topics
        topic_vectors_2d = umap_model.transform(topics_info['topic_vectors'])
        topics_idx_vector = []
        for idx, vectors in zip(topics_info['topic_idxes'], topic_vectors_2d):
            topics_idx_vector.append({'id': idx,
                                    'x': vectors[0],
                                    'y': vectors[1]
            })
        
        # docs
        docs_idx_vector = []
        document_ids = self.document_ids
        topic_indice, _, _, _ = super().get_documents_topics(document_ids, reduced=True) if is_reduced \
                     else super().get_documents_topics(document_ids)
        for doc_id, vectors, topic_idx in zip(document_ids, document_vectors_2d, topic_indice):
            #doc_id = super()._get_document_ids(idx)
            docs_idx_vector.append({'id': doc_id,
                                    'x': vectors[0],
                                    'y': vectors[1],
                                    'topic_idx': topic_idx,
            })

        words_idx_vector = []
        total_topic_words = set(word for words in topics_info['topics_words'] for word in words)
        word_vectors_2d = umap_model.transform(super()._get_word_vectors(total_topic_words))
        for word, vectors in zip(total_topic_words, word_vectors_2d):
            words_idx_vector.append({'id': self.word2index[word],
                                    'x': vectors[0],
                                    'y': vectors[1],
                                    'word': word
            })

        return topics_idx_vector, docs_idx_vector, words_idx_vector

    def get_links_info(self, word_num=10, is_reduced=None):
        if is_reduced is None:
            is_reduced = self.is_reduced

        links = []

        
        document_ids = self.document_ids
        topic_indexes, _, _, _ = super().get_documents_topics(document_ids, reduced=True) if is_reduced \
                     else super().get_documents_topics(document_ids)

        #doc -> topic
        links += [('doc_' + str(doc_id), 'topic_' + str(topic_idx)) for doc_id, topic_idx in zip(document_ids, topic_indexes)]


        # topic-word
        topics_words = self.get_topics_info(is_reduced=is_reduced)['topics_words']
        links += [('word_' + str(self.word2index[word]), 'topic_' + str(topic_idx)) for topic_idx, words in enumerate(topics_words) for word in words[:word_num]]

        # doc-word
        #print(self.get_keywords_by_docs([document_ids[0]])[0])
        docs_words, _ = self.get_keywords_by_docs(document_ids)
        print(docs_words)
        links += [('doc_' + str(doc_id), 'word_' + str(self.word2index[word])) for doc_id, words in zip(document_ids, docs_words) for word in words[:word_num]]
        
        return links