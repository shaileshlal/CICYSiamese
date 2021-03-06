# Clustering Analyses for the CICY Configuration Matrices
We use K means clustering and K nearest neighbors classification to infer the map between CICY configuration matrices
and the Hodge number _h<sup>1,1</sup> before passing the matrices through the Siamese Network. 

This is mostly illustrative and for comparison with the Siamese Network analysis in the notebooks [here](../SiameseClustering)

1. [edaCICY3kmeans](edaCICY3kmeans.ipynb):

This contains an initial application of K-means clustering to the CICY3 dataset. We use inertia and silhouette score to fix on optimal
values of _k_ for each _h<sup>1,1</sup> subcluster, and also visualize a putative similarity score between two CICY3 manifolds.

3. [naiveCICY3kNN](naiveCICY3kNN.ipynb):

This contains a k-nearest neighbors analysis of the CICY3 dataset using the configuration matrices themselves as input data. This gives
us a baseline to compare the results obtained by using the embedding learnt by the Siamese network. We reach an accuracy of 56% by training
on a comparable amount of data.
