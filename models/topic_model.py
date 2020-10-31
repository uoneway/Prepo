from submodules.Top2Vec.top2vec import Top2Vec

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
        self.topics_info = []  # 초기화 
        for topic_idx, topic_words in zip(topic_indice, topics_words):
            self.topics_info.append({'topic_idx': topic_idx,
                                'topic_words': topic_words
            })


    def get_doc_ids_by_topic(self, topic_idx):
        if self.is_reduced:
            num_docs = super().get_topic_sizes(reduced=True)[0][topic_idx]
            docs, doc_scores, doc_ids = super().search_documents_by_topic(topic_idx, num_docs, reduced=True)
        else:
            num_docs = super().get_topic_sizes()[0][topic_idx]
            docs, doc_scores, doc_ids = super().search_documents_by_topic(topic_idx, num_docs)
        return doc_ids


    # def get_reduced_cluster_idx(self, cluster_idx, clusters_reduced_indice_list):
    #     for idx, sub_cluster in enumerate(clusters_reduced_indice_list):
    #         if cluster_idx in sub_cluster:
    #             return idx
    #     print("ERROR: get_reduced_cluster_idx")

    def reduce_topic(self, num_topics):
        # 줄이고 싶다면 이 함수를 무조건 호출해줘야함 
        clusters_reduced_indice_list = self.hierarchical_topic_reduction(num_topics)
        self.is_reduced = True

        self.update_topics_info()

        # for doc_id, cluster_idx in self.doc_to_cluster.items():
        #     self.doc_to_cluster[doc_id]['cluster_reduced']= self.get_reduced_cluster_idx(cluster_idx, clusters_reduced_indice_list)



    # def search_documents_by_topic(self, topic_num, num_docs, return_documents=True, reduced=False):
    # def search_documents_by_keywords(self, keywords, num_docs, keywords_neg=None, return_documents=True):
    # def search_documents_by_documents(self, doc_ids, num_docs, doc_ids_neg=None, return_documents=True):

    # get_documents_topics(self, doc_ids, reduced=False)
