import streamlit as st
import pandas as pd


@st.cache_data
def load_movies():
    return pd.read_parquet("demo/movies.parquet")


@st.cache_data
def load_result():
    return pd.read_parquet("demo/result.parquet")


movies = load_movies()
result = load_result()

"""
# Recommendation Result Demo Page

Hyperparameter combination that draws following metrics from MovieLens 20M data is empirically selected to be optimal.
For more details, please refer to [hyperparameter optimization](/hyperparameter_optimization) page. 
"""

test_metrics = {
    "precision@20": 0.0859,
    "recall@20": 0.2209,
    "hitRatio@20": 0.8034,
    "ndcg@20": 0.0921,
}
st.json(test_metrics)

st.button("✅ click to sample another result")
sample = result.sample(1).to_dict("records")[0]

st.header("Sampled user's holdout watch history")
true = sample.pop("target")
true = movies.loc[movies["movie_id"].isin(true)]
true.insert(1, "matched", [1] * len(true))
st.dataframe(true.drop(["movie_id", "matched"], axis=1).reset_index(drop=True))

st.header("Recommended movies")
pred = sample.pop("prediction")
pred = movies.loc[movies["movie_id"].isin(pred)]
st.dataframe(
    pred.set_index("movie_id").join(
        true.drop(["title", "genres"], axis=1).set_index("movie_id")
    )
)
