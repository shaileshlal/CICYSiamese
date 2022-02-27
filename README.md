# Few Shot Learning Complete Intersection Calabi Yau manifolds using Siamese Neural Networks

This is a repository of codes for and around our work in [arXiv:2111.04761](https://arxiv.org/abs/2111.04761).

In this paper we apply Siamese Neural Networks to learn a similarity score for Complete Intersection Calabi Yau manifolds of complex
dimension 3. These datasets are known as CICY3 in the literature, and are downloadable [here](http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/)
and [here](http://www1.phys.vt.edu/cicydata/).

These are a class of manifolds which are of central importance for building string theory models 
explaining the physics of our four dimensional universe.

String theory is defined in 10 spacetime dimensions (1 time + 9 space) and our universe as we know it is defined in
4 dimensions (1 time + 3 space). The missing 6 dimensions may be naturally accounted for by postulating that they
are rolled up tightly and are hence invisible to current experiments. Further, the geometry of the missing 6 dimensions 
dictates the physics of the observable 4 dimensions. 

Even at a very preliminary level CICY3 manifolds have 3 complex (and hence 6 real) space dimensions and as are suitable 
for such model building. Their additional mathematical properties make them even more compelling choices.

## Overview of Dataset

The CICY3 dataset consists of configuration matrices, which are two-dimensional tensors along with the number of
rows and columns of each matrix as input data. The target variables are the topological data of the 
CICY3. These are two positive integers, the Hodge numbers _h<sup>1,1</sup>_, _h<sup>2,1</sup>_ 
and the second Chern class _C<sub>2</sub>_, an array of positive integers.

The statistics of these input and target variables are explored [here](ExploratoryDataAnalyses/).

The configuration matrices are the independent variables from which the above topological 
data may be computed. This is a complicated numerical problem, a modern treatment is 
[here](https://arxiv.org/abs/0805.2875).

A considerable amount of work has gone into using machine learning to predict the Hodge 
numbers of the CICY3 manifolds by `looking' at the configuration matrices, much as
ML can infer the value of a handwritten digit by looking at its picture. The state of the
art is [here](https://arxiv.org/abs/2007.13379).

Knowledge of this topological data is crucial to using these manifolds to build models of the four dimensional
universe. For example, this data controls the number of elementary particles visible in the four dimensional universe.

## The Landscape Problem and Similarity
The possible choices for the missing six dimensions in string theory are not restricted to CICY3. Estimates for the full number
range from 10<sup>500</sup> to 10<sup>278,000</sup>. This _landscape_ is an impossibly large number among which one has to search for our universe
(in comparison, [ImageNet](https://www.image-net.org) has about 10<sup>7</sup>
elements and there are about 10<sup>80</sup> atoms in the universe). _A priori_, the
search can only proceed by explicitly computing the four dimensional physics for each choice, which can be prohibitively
difficult even for a single example. This is the _string landscape problem_.

If one could organize the landscape according to similarity, so that manifolds that give rise to similar four-dimensional physics
are clustered together then given a new candidate manifold we could evaluate its likelihood of giving rise to a particular kind of
four-dimensional physics by examining which cluster this new manifold is predicted to lie in.

## Our Analysis

We use Siamese Networks to infer similarity across this dataset and cluster similar manifolds together. As an added bonus, since Siamese Networks
are useful for _few-shot learning_, we only need to train on about 1 percent of the available dataset. Our analysis shows that

1. The clustering learnt by the Siamese Network is very precise. By using a K nearest neighbours classifier on the clustered data we can
achieve an accuracy of 98.2 \% on the test set.
2. More generally, by training on very few data and extrapolating across the whole dataset explicitly show how machine learning may be used to
address the string landscape problem even in the face of very limited available data. 

The second point is especially relevant to the current state of the art as we indeed have very limited data from the string landscape, and the
overwhelmingly vast majority of it is completely unexplored (i.e. unlabelled).

## Project Structure

The project is organized into several Jupyter notebooks in subfolders. They are written in Python3 and use Tensorflow/Keras and scikit-learn, numpy
and pandas as the major tools.

1. [ExploratoryDataAnalyses](ExploratoryDataAnalyses/): contains EDA for the CICY3 dataset. This is not directly used, but does contain some illustrative
observations.
2. [NaiveClustering](NaiveClustering/): using K means clustering and K nearest neighbors on the CICY3 dataset without passing it through a Siamese Network.
This is again mostly illustrative and for comparison with the Siamese Network analysis.
3. [SiameseNetwork](SiameseNetwork/): training the Siamese Network on the CICY3 dataset. Visualizing the learnt clustering in various ways.
4. [SiameseClustering](SiameseClustering/): using K means clustering and K nearest neighbors on the embedding learnt by the Siamese Network [here](SiameseNetwork/)
to draw inferences across the test data.
5. [UtilityScripts](UtilityScripts/): some helper scripts for handling the CICY3 dataset.

This analysis results in the following representation of the CICY3 dataset as a scatterplot. Similar manifolds are clustered together.
![the CICY3 dataset clustered by similarity](SiameseNetwork/cicy3.png)
*Visualizing the CICY3 dataset clustered by similarity. Colors encode h<sup>1,1</sup>. Similar manifolds have the same h<sup>1,1</sup>.*

