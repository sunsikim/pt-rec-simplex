import streamlit as st
import os

"""
# SimpleX Model Architecture

This page briefly summarizes original SimpleX [paper](https://arxiv.org/abs/2109.12613) to introduce the meaning and role
of each hyperparameter in the model. The paper first insists that recent researches on collaborative filtering algorithm 
tend to focus only on *interaction encoder* aspect and neglect two other critical building blocks: *loss function* and 
*negative sampling strategy*. 

Among these two relatively untouched components, this paper contributes to draw attention on loss function by introducing 
novel objective called **Cosine Contrastive Loss**(or, CCL). Authors succeeded to prove the effectiveness of this new
loss function by reaching SOTA metrics on widely used evaluation datasets like AmazonBooks with relatively simple 
architecture that the authors introduces in this paper.

So, to fully comprehend the algorithm, (1) SimpleX model architecture and (2) CCL has to be explained.

## 1. SimpleX Model Architecture

"""

st.image(f"{os.getcwd()}/demo/images/architecture.png")
st.caption("illustration of the architecture provided in the paper")

"""
* Each one of $U$ users and $I$ items is represented by $d$-dimensional vectors.
* Paper introduces three different methods to aggregate the user history on interacted items, but average pooling is 
showed to be simple and effective.
* Denote $e_u$ as user vector of $u$-th user and $p_u$ as average pooled item embeddings in the user's interaction history.
Then weighted sum of $e_u$ and $p_u$(denote as $h_u$) is calculated as $h_u=ge_u+(1-g)Vp_u$ where $V$ is $(d,d)$ 
shaped dense layer and $g$ is configurable scalar weight on user embedding.
* cosine similarity between $h_u$ and $e_i$(vector of $i$-th item) is then calculated to be fed into CCL.

## 2. Cosine Contrastive Loss(CCL)

Denote the cosine similarity between $h_u$ and $e_i$ as $\\hat{y}_{ui}$. Then CCL is defined as

$$
\\mathcal{L}_{CCL}(u,i)=(1-\\hat{y}_{ui})+\\frac{w}{|\\mathcal{N}|}\\sum_{j\\in\\mathcal{N}}\\max(0,\\hat{y}_{uj}-m)
$$

where $\\mathcal{N}$ denotes set of negative samples for product $i$ and $|\\mathcal{N}|$ is its cardinality(i.e. negative 
sample size). This formula can be decomposed into positive loss and negative loss part; $(1-\\hat{y}_{ui})$ and the other.

* postive part : Loss function should reward $\\hat{y}_{ui}$ close to 1, since this is cosine similarity between user $u$ 
and interacted items $i$. 
* negative part : It should penalize high value of $\\hat{y}_{uj}$, so the sign of it is positive. However, it only 
accumulates loss from negative samples whose cosine similarity between $h_u$ is larger than a margin $m$. That is, if 
model does so well in distinguishing negative sample that it calculates $\\hat{y}_{uj}$ to be less than $m$, it is not 
counted as loss(as paper describes, CCL can 'automatically filter out hard negative samples that are hard to distinguish').

$w$ is another hyperparameter that controls the weight on the negative loss. However, overall scale of this value would be 
dependent on another controlled variable, number of negative samples $|\\mathcal{N}|$, so ignored in this experiment. 
"""
