# Few Shot Learning Complete Intersection Calabi Yau manifolds using Siamese Neural Networks

This is a repository of codes for and around our work in [arXiv:2111.04761](https://arxiv.org/abs/2111.04761)

In this paper we apply Siamese Neural Networks to learn a similarity score for Complete Intersection Calabi Yau manifolds of complex
dimension 3. This dataset is known as CICY3 in the literature, and is downloadable [here](http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/).

These are a class of manifolds which are of great relevance when building models that contain the physics of our four dimensional universe.

In addition, they are perhaps the first dataset created for Algebraic Geometry and String Theory.

The CICY3 dataset consists of configuration matrices, the second chern class $C_2$, 
the Hodge numbers, the number of polynomials and the number of projective spaces. 

The topological data is contained in the Hodge numbers and the second Chern class.
The more familiar Euler number is determined as a linear combination of the Hodge
numbers. 

The configuration matrices are the independent variables from which the above topological 
data may be computed. This is a complicated numerical problem, a modern treatment is 
[here](https://arxiv.org/abs/0805.2875).

A considerable amount of work has gone into using machine learning to predict the Hodge 
numbers of the CICY3 manifolds by `looking' at the configuration matrices, much as a
ML algorithm can infer the value of a handwritten digit by looking at its picture.
