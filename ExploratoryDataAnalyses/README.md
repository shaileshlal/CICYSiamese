# Exploratory Data Analyses for the CICY3 Dataset

This is an overview of some exploratory data analyses for the CICY3 dataset. 
and takes the form of configuration matrices $Q$, the second chern class $C_2$, 
the number of polynomials and the number of projective spaces. 

Among these, the configuration matrices are the independent variables on the 
basis of which we would like to predict topological data of the CICY3 manifolds.

The topological data is contained in the variables $h^{1,1}$, $h^{2,1}$ and $C_2$.
Further, the Euler number $\chi$ is determined in terms of the Hodge numbers \textit{via}
\begin{equation}
\chi = 2\left(h^{1,1}-h^{2,1}\right)
\end{equation}
## Explanation of the keys:

**c2**: vector of integers, length varies from 15 to 1. Entries are sampled from {0, 24, 36, 40, 44, 48, 50, 52, 54, 56, 60, 64, 72}

**favour**: If the data entry is _favourable_ i.e. h11 = num_cp. Binary, 1 if True

**h11**: Hodge number

**h21**: Hodge number

**isprod**: is the manifold a direct product of lower dimensional manifolds

**kahlerpos**:

**matrix**: the _configuration matrix_ $Q$ which we treat as analogues of MNIST images (say)

**size**: shape of the configuration matrix. Of the form (num_cp,num_eqs) defined below

**euler**: Euler number $\chi = 2\left(h^{1,1}-h^{2,1}\right)$

**num_cp**: number of projective spaces (number of rows of $Q$)

**num_eqs**: number of equations (number of columns of $Q$)

**dim_cp**: list of projective space dimensions

**min_dim_cp**: statistics of dim_cp; min

**max_dim_cp**: statistics of dim_cp; max

**mean_dim_cp**: statistics of dim_cp; mean

**median_dim_cp**: statistics of dim_cp; median

**num_dim_cp**:

**num_cp_1**: number of $\mathbb{P}^1$

**num_cp_2'**: number of $\mathbb{P}^2$

**num_cp_neq1**: number of $\mathbb{P}^1$ with $n\neq 1$.

**num_over**:

**num_ex**: the excess number $N_{ex} =\sum_{r=1}^F\left(n_r+f+m-2k\right)$

**deg_eqs**: list of equation degrees 

**min_deg_eqs**: min of deg_eqs

**max_deg_eqs**: max of deg_eqs

**mean_deg_eqs**: mean of deg_eqs

**median_deg_eqs**: median of deg_eqs

**num_deg_eqs**: number(?) of equation degrees

**rank_matrix**: the rank of the configuration matrix

**norm_matrix**: the Froebenius norm of the configuration matrix

**dim_h0_amb**: dimension of the homology group of the ambient space
