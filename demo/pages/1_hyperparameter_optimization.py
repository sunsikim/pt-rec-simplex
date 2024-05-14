import streamlit as st
import os

st.title("Hyperparameter Optimization")

"""
This page compares 4 validation metrics(precision, recall, hit ratio, ndcg at 20) calculated when using each of 
hyperparameter combination applied on MovieLens 1M dataset. When effect of each of four hyperparameters is evaluated, 
rest are fixed to control the effect of tuning other hyperparameters and only observe the marginal effect of it. Image
captured from Tensorboard are followed by brief caption to illustrate the effect of tuning each hyperparameter.
"""

st.header("1. Effect of batch size")

st.image(f"{os.getcwd()}/demo/images/hpo_batch.png")

st.text("✅ smaller the batch size, the better metrics.")

st.header("2. Effect of embedding dimension")

st.image(f"{os.getcwd()}/demo/images/hpo_dim.png")

st.text("✅ larger the embedding dimension, the better metrics.")

st.header("3. Effect of negative loss margin(*m*) in CCL")

st.image(f"{os.getcwd()}/demo/images/hpo_margin.png")

st.text("✅ relationship between metrics and margin value is not monotonic.")

st.header("4. Effect of negative samples size")

st.image(f"{os.getcwd()}/demo/images/hpo_negsize.png")

st.text("✅ relationship between metrics and negative sample size is not monotonic.")

"""
Based on these results, baseline value of hyperparameters are set to be following when training a model with MovieLens 
20M dataset.

* `batch_size` = 16
* `embedding_dim` = 200
* `loss_negative_margin` = 0.3
* `negative_sample_size` = 600 
"""
