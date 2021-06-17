# Similarity Network Fusion
---
## Background
#### Overall
  - multiple and diverse **genome-scale data**
  - **integrative methods** are essential for capture biological information
  - identification of homogeneous **subtypes** in one cancer
#### Integrative challenge
 
 - the small number of **samples** compared to the large number of measurements.
 - the **differences** in scale, collection bias and noise in each data set
 - the **complementary nature** of the information provided by different types of data

#### Former methods
- **concatenate** normalized measurements
      *low signal-noise ratio*
- analyze each data type **independently** before combining
      *hard to integrate*
- **preselect** important genes and use Consensus Clustering
      *biased analysis*

## SNF
#### Main steps
- **construct** a sample-similarity network for each data type
- **integrate** these networks into a single similarity network using a nonlinear combination method
![SNF steps](1.png)
#### Advantages
- capture both **shared** and **complementary** information, offered insight into how informative each data type is to the observed similarity
- derive useful information from a **small** number of samples
- **robust** to noise, data heterogeneity and scales to a large number of genes
- make **efficient** identifies of subtypes among existing samples by clustering and predict labels for new samples
#### Details
- A **patient similarity network** is represented as a graph $G=(V,E)$  
  - $V = \{x_1, x_2, x_3, ..., x_n\}$ correspond to the patients  
  - $E$ similarity between patients
  - $W$ is the similarity matrix
  $$
  \begin{aligned}
  \rho(x_i, x_j) &= Euclidean \space distance \space  between \space  x_i \space and \space x_j  \\
  \epsilon_{i,j} &= \frac{mean(\rho(x_i, N_i)) + mean(\rho(x_j, N_j))+\rho(x_i, x_j)}{3} \\
  W(i, j) &= exp(-\frac{\rho^2(x_i,x_j)}{\mu\epsilon_{i,j}})
  \end{aligned}
  $$
- Define a full and sparse **kernel** to normalize weighted matrix on the vertex $V$
  - full kernel is a normalized weight matrix $P$
  $$
  \begin{aligned}
   P &= D^{-1}W \\
   D(i, i) &= \sum_{j}W(i,j) \\
   \sum_{j}P(i,j) &= 1
  \end{aligned}
  $$
  - a little modification to eliminate self-similarities
  $$
  \begin{aligned}
   P(i,j) &= \begin{cases}
   \frac{W(i,j)}{2\sum_{k\not ={i}}{W(i, k)}}, \small{j\not ={i}} \\
   \frac{1}{2}, \small{j=i}
   \end{cases}   
  \end{aligned}
  $$
- use KNN to measure local **affinity** (non-neighboring points)
  $$
  S(i,j) = \begin{cases}
  \frac{W(i,j)}{\sum_{k\in{N_i}W(i,k)}}, \small{j\in{N_i}} \\
  0, \small{otherwise}
  \end{cases}
  $$
- iteratively **update** similarity matrix corresponding to each of the data types
  $$
  \begin{aligned}
  \textbf{P}^{(v)} = \textbf{S}^{(v)} \times \Big(\frac{\sum_{k\not ={v}}\textbf{P}^{k}}{m-1}\Big) \times (\textbf{S}^{(v)})^{T} 
  \end{aligned}
  $$
- example: two data types
  $$
  \textbf{P}^{(1)}_{t+1} = \sum_{k\in N_i} \sum_{l \in N_j} \textbf{S}^{(l)}(i,k) \times \textbf{S}^{(l)}(j,l) \times \textbf{P}^{(2)}_{t}(k,l)
  $$
  - similarity information is only propagated through the common neighborhood
  - comlementary information from other data type

## WSNF
#### Background
- Existing methods rarely use information from gene regulatory networks to facilitate the subtype identification. In other words, the information among features is ignored.
#### Main steps
- Constructe the regulatory netword
  ![regulatory network](2.png)
- Calculate feature weights
- Weighted similarity network fusion
#### Advantage
- Make use of both the expression data and network information. Take the feature weight into consideration, so perform better than SNF.
#### Details
- Compute ranking of features using Google PageRank
  Network is defined as $G(V, E)$. The nodes $V$ are the features, and the edges $E$ are the interactions. The direciton of an edge is from a regulator to its target.
  $$
  \begin{aligned}
  N features &= \{f_1, f_2, ..., f_N\} \\
  R(f_i) &= \frac{1-d}{N} + d \sum_{f_j \in T(f_i)} \frac{R(f_j)}{L(f_i)}
  \end{aligned}
  $$
  Normalize the ranks as:
  $$
  R_N(f_i) = \frac{R(f_i)}{\sum_{m=1}^{N}R(f_m)}
  $$
- Integrate feature ranking and feature variantion
  - $X(f_i)$ is a numeric vector representing the expression value of feature $f_i$ across all samples
  The MAD(median absolute deviation) of a feature $f_i$ is calculated as:
  $$
  MAD(f_i) = median(|X(f_i) - median(X(f_i))|)
  $$  
  Normalize the MADs as:
  $$
  MAD_N(f_i) = \frac{MAD(f_i)}{\sum_{m=1}^{N}MAD(f_m)}
  $$
  - Apply a linear model to integrate these two measures to get the final weight
    $$
    W(f_i) = \beta R_N(f_i) + (1-\beta)MAD_N(f_i)
    $$
- Weighted similarity network fusion
  $$
  Distance(S_i, S_j) = \sqrt{\sum_{m=1}^{P}W(f_m) * (f_m^{S_i}-f_m^{S_j})^2} \space \forall i,j \le n, i\not = j
  $$
- Execute SNF algorithm