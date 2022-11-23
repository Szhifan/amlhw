
import numpy as np 
import pandas as pd 
import pickle
import seaborn as sns 
import re 
import random
from collections import Counter
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
path="df_data/phrases_data2.df"
features_path="features/features.txt"
neg_feature_path="features/neg_features.txt"
df=pickle.load(open(path,"rb"))

def add_neg(ph:str):
    """
    This function adds "neg_" suffix to each word in the phrase when negation words: not 
    are detected.  
    """
    res=""
    if re.search(r"(\snot\s)|(^not\s)|(\snever\s)|(^never\s)|((\s*).+n't)",ph):
        
        tokens=word_tokenize(ph)
        n=len(tokens)
        pos=2**32-1 
        for i in range(n):
            
            if re.search(r"not|.*n't",tokens[i]):
                pos=i 
            if i>pos:
                res+="neg_{} ".format(tokens[i])
            else:
                res+=tokens[i]+" "

        return res 
    return ph 
def bi_feature_vec(ph,features):
    """
    returns the feature vector of a phrase. The returned object is a np.array with binary
    values, each value correspond to the presence of the feature word in the phrase. 
    """
    tokens=word_tokenize(ph)
    n=len(features)
    vec=np.zeros(n)
    for i in range(n):
        if features[i] in tokens:
            vec[i]=1 
    return vec 
# features=open(features_path,"r").read().split("\n")[:-1]
# neg_features=open(neg_feature_path,"r").read().split("\n")[:-1] 
# vectors=[]
# kws=df["neg_keywords"]
# for kw in kws:
#     vec=bi_feature_vec(kw,neg_features)
#     vectors.append(vec)
# vectors=np.array(vectors)
# np.save("features/neg_vec3.npy",vectors)

# zerovec_class=[0]*5
# for i in range(len(v)):
#     print(i)
#     if np.sum(v[i])==0:
#         label=int(df2.loc[i]["label"]) 
#         if label==1:
#             zerovec_class[label-1]+=1 
#         if label==2:
#             zerovec_class[label-1]+=1 
#         if label==3:
#             zerovec_class[label-1]+=1 
#         if label==4:
#             zerovec_class[label-1]+=1 
#         if label==5:
#             zerovec_class[label-1]+=1 
# labels=list(range(1,6))
# sns.barplot(labels,zerovec_class)   
# plt.show()
# plt.savefig("figures/zerovec_dis.png")    
    
# v=np.delete(v,drop_index,0)
# v2=np.delete(v2,drop_index,0)
# print(v.shape==v2.shape)
# df2=df2.drop(df2.index[drop_index]) 
# pickle.dump(df2,open("df_data/phrases_data2.df","wb"))
# np.save("features/vec2.npy",v)
# np.save("features/neg_vec2.npy",v2)
