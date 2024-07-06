import os
import json
from incdbscan import IncrementalDBSCAN
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import rand_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer



class Modelling:

    def __init__(self):
        self.model=IncrementalDBSCAN(eps=0.25, min_pts=7)
        self.labels=[]
        self.label_names_map={}
        self.label_sematics_map={}
        self.cluster_labels=[]

    @classmethod
    def load(cls,file_name):
        obj = load(file_name)
        return obj
    
    def save(self,model,file_name):
        dump(model,file_name)
        
    def get_embeddings(self,data):
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(data)
        return embeddings

    def get_sematic_words(self,data):
        try:
            count_model = CountVectorizer()
            count_model.fit_transform(data)
            return count_model.get_feature_names_out()
        except:
            return []
    
    def insert(self,data):
        embeddings = self.get_embeddings(data)
        try:
            self.model.insert(embeddings)
        except:
            return "error encountered"

        self.cluster_labels = [int(x) for x in self.model.get_cluster_labels(embeddings)]
        tmp_labels = list(set(self.cluster_labels))

        if(len(tmp_labels)!=self.labels):
            #self.labels = list(set(cluster_labels))
            doc_label_map={x:[] for x in tmp_labels}
            for i in range(len(self.cluster_labels)):
                doc_label_map[self.cluster_labels[i]].append(data[i])
            
            tmp_label_sematics_map={i:self.get_sematic_words(doc_label_map[i]) for i in tmp_labels}
            label_names = get_label_names_llm(list(tmp_label_sematics_map.values()))
            print(tmp_labels)
            print(label_names)
            tmp_label_names_map = {tmp_labels[i]:label_names[i] for i in range(len(tmp_labels))}

            if(not self.labels):
                self.labels = tmp_labels
                self.label_sematics_map = tmp_label_sematics_map
                self.label_names_map = tmp_label_names_map

            else:
                for i in tmp_labels:
                    if( i not in self.labels):
                        self.labels.append(i)
                        self.label_sematics_map[i] = tmp_label_sematics_map[i]
                        self.label_names_map = tmp_label_names_map[i]

            return "Model Updated Sucessfully, Clusters Changed"
        
        else:

            return "Model Updated Successfully"
