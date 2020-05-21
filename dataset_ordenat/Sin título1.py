# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:35:21 2020

@author: Caterina
"""

b = np.array(
          [[0,0,0,0,0,1] * n_anger_tr]+ [[0,0,0,0,1,0] *n_disgust_tr ]
        + [[0,0,0,1,0,0]*n_fear_tr]
        + [[0,0,1,0,0,0]*n_happiness_tr]
        + [[0,1,0,0,0,0]*n_sadness_tr] +
          [[1,0,0,0,0,0]*n_surprise_tr])

b.shape
b


nb_train_samples = 3926
a = np.zeros((nb_train_samples,6))
print(a)
i=0
a[i:n_anger_tr,1]=1
i+=n_anger_tr
print(a,"\n")

a[i:n_disgust_tr,2]=1
i+=n_disgust_tr
print(a,"\n")

a[i:n_fear_tr,3]=1
i+=n_fear_tr
print(a,"\n")

a[i:n_happiness_tr,4]=1
i+=n_happiness_tr
print(a,"\n")

a[i:n_sadness_tr,5]=1
i+=n_sadness_tr
print(a,"\n")

a[i:n_surprise_tr,6]=1
print(a,"\n")
