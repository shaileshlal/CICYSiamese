# Clustering Analyses on the Embedding Representation Learnt by the SNN

The siamese network learns a representation of the CICY3 data as points in an embedding space, with manifolds 
having the same value being clustered together.

We show two immediate applications of the embedding representation learnt by the SNN for the CICY3 dataset.
In both cases we use clustering analyses analogous to those [done previously without the SNN](../NaiveClustering).

1. We use K nearest neighbors classification to [infer the correct _h<sup>1,1</sup>_](siameseCICY3knn.ipynb) value of manifolds in the test
set by examining which cluster they belong to in embedding space.

2. We also use this embedding representation to characterize [typicality in the CICY3 data](typicalCICY3kmeans.ipynb). 
Intuitively, an instance of data is typical if it is 'like most of the previous instances'. We show how the
representation learnt by the Siamese Network ties in with this intuitive notion of typicality.
