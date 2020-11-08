from submodules.Top2Vec.top2vec import Top2Vec
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
        
        self.umap_args = {'n_neighbors': 3,
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

        self.topics_info = []
        self.update_topics_info()

        
    def get_topics_info(self):
        return self.topics_info
       
        
    def update_topics_info(self):
        topics_words, _, topic_indice = self.get_topics(reduced=True) if self.is_reduced \
                                else self.get_topics() 
        topic_vectors = self.topic_vectors_reduced if self.is_reduced else\
                        self.topic_vectors

        self.topics_info = []  # 초기화 
        for topic_idx, topic_words, topic_vector in zip(topic_indice, topics_words, topic_vectors):
            self.topics_info.append({'topic_idx': topic_idx,
                                'topic_words': topic_words,
                                'topic_vector': topic_vector,
            })


    def get_doc_ids_by_topic(self, topic_idx):
        if self.is_reduced:
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


    def get_keywords_by_doc(self, doc_ids, doc_ids_neg=None,):
    
        # (self, doc_ids, num_words, doc_ids_neg=None,):
        if doc_ids_neg is None:
            doc_ids_neg = []

        self._validate_doc_ids(doc_ids, doc_ids_neg)

        doc_indexes = self._get_document_indexes(doc_ids)
        doc_indexes_neg = self._get_document_indexes(doc_ids_neg)

        doc_vecs = [self.document_vectors[ind] for ind in doc_indexes]
        doc_vecs_neg = [self.document_vectors[ind] for ind in doc_indexes_neg]
        combined_vector = self._get_combined_vec(doc_vecs, doc_vecs_neg)

        # 이름은 topic word 찾는거지만, 벡터받아서 가까운 50개 단어 추출해줌
        combined_vector = np.expand_dims(combined_vector, axis=0)  # 1d라서 2d로 바꿔주기
        words, word_scores = super()._find_topic_words_and_scores(combined_vector)

        return words, word_scores

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

    def get_2d_vectors(self):
        # args for UMAP. Same as current args except n_components to plot graph
        umap_args_for_plot = self.umap_args.copy()
        umap_args_for_plot.update({'n_components': 2,})

        # Dimension reduction
        umap_model = umap.UMAP(**umap_args_for_plot, random_state=42).fit(super()._get_document_vectors())  #  + self.topic_words
        document_vectors_2d = umap_model.embedding_  # same as umap_model.transform(self._get_document_vectors())

        #topics
        total_topic_words = set()
        topic_vectors = []
        topic_idx = []
        for info in self.get_topics_info():
            topic_idx.append(info['topic_idx'])
            total_topic_words.update(info['topic_words'])
            topic_vectors.append(info['topic_vector'])
        topic_vectors_2d = umap_model.transform(topic_vectors)
        word_vectors_2d = umap_model.transform(super()._get_word_vectors(total_topic_words))
        # super().document_id
        # 
        topics_idx_vector = []
        for idx, vectors in zip(topic_idx, topic_vectors_2d):
            topics_idx_vector.append({'id': idx,
                                    'x': vectors[0],
                                    'y': vectors[1]
            })
        
        docs_idx_vector = []
        document_ids = self.document_ids
        topic_indice, _, _, _ = super().get_documents_topics(document_ids, reduced=True) if self.is_reduced \
                     else super().get_documents_topics(document_ids)
        for doc_id, vectors, topic_idx in zip(document_ids, document_vectors_2d, topic_indice):
            #doc_id = super()._get_document_ids(idx)
            docs_idx_vector.append({'id': doc_id,
                                    'x': vectors[0],
                                    'y': vectors[1],
                                    'topic_idx': topic_idx,
            })
        words_idx_vector = []
        for word, vectors in zip(total_topic_words, word_vectors_2d):
            words_idx_vector.append({'id': self.word2index[word],
                                    'x': vectors[0],
                                    'y': vectors[1]
            })

        return topics_idx_vector, docs_idx_vector, words_idx_vector


    def reduce_topic(self, num_topics):
        # 줄이고 싶다면 이 함수를 무조건 호출해줘야함 
        clusters_reduced_indice_list = self.hierarchical_topic_reduction(num_topics)
        self.is_reduced = True

        self.update_topics_info()

        # for doc_id, cluster_idx in self.doc_to_cluster.items():
        #     self.doc_to_cluster[doc_id]['cluster_reduced']= self.get_reduced_cluster_idx(cluster_idx, clusters_reduced_indice_list)



    ## topic, docs, keyword, 
    # topic -> docs
    # def search_documents_by_topic(self, topic_num, num_docs, return_documents=True, reduced=False):
    # topic -> keyword
    # def get_topics(self, num_topics=None, reduced=False):
    # return self.topic_words[0:num_topics], self.topic_word_scores[0:num_topics], np.array(range(0, num_topics))

    # keywords -> doc
    # def search_documents_by_keywords(self, keywords, num_docs, keywords_neg=None, return_documents=True):
    # keywords -> keywords
    # def similar_words(self, keywords, num_words, keywords_neg=None):
    # keywords -> topic
    # def search_topics(self, keywords, num_topics, keywords_neg=None, reduced=False):

    # docs -> topics
    # get_documents_topics(self, doc_ids, reduced=False)
    # docs -> keyword
    # docs -> docs
    #  def search_documents_by_documents(self, doc_ids, num_docs, doc_ids_neg=None, return_documents=True):
