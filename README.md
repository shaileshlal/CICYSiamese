# Few Shot Learning Complete Intersection Calabi Yau manifolds using Siamese Neural Networks

This is a repository of codes for and around our work in [arXiv:2111.04761](https://arxiv.org/abs/2111.04761)

In this paper we apply Siamese Neural Networks to learn a similarity score for Complete Intersection Calabi Yau manifolds of complex
dimension 3. This dataset is known as CICY3 in the literature, and is downloadable [here](http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/)
and [here](http://www1.phys.vt.edu/cicydata/).

These are a class of manifolds which are of central importance for building string theory models 
explaining the physics of our four dimensional universe.


## Overview 

The CICY3 dataset consists of configuration matrices, which are $2d$ tensors along with the number of
rows and columns of each matrix as input data. The target variables are the topological data of the 
CICY3. These are two positive integers, the Hodge numbers $h^{1,1}$ and $h^{2,1}$ and the second Chern
class, an array of positive integers.

The statistics of these input and target variables are explored [here](ExploratoryDataAnalyses/).

The configuration matrices are the independent variables from which the above topological 
data may be computed. This is a complicated numerical problem, a modern treatment is 
[here](https://arxiv.org/abs/0805.2875).

A considerable amount of work has gone into using machine learning to predict the Hodge 
numbers of the CICY3 manifolds by `looking' at the configuration matrices, much as a
ML algorithm can infer the value of a handwritten digit by looking at its picture.

Our analysis uses Siamese Networks to infer similarity across this dataset
