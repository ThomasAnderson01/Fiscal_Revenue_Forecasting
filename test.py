# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:42:20 2019

@author: chern.lei
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=1)
pca.fit(X)  

pca.transform(X)
print(pca.explained_variance_ratio_)  

print(pca.singular_values_)





