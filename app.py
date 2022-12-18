"""
# Mastodon stats
Visualizing Mastodon instances
"""

import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

from api import Instance, fetch_timeline
from api import instance_info
from neighboorhood_to_web import convert_for_web

st.set_page_config(layout="wide")

instance = Instance("sigmoid.social")


@st.cache
def fetch_info():
    def _format(info):
        return {
            "title": info["title"],
            "description": info["description"],
            "user_count": info["stats"]["user_count"],
            "status_count": info["stats"]["status_count"],
            "domain_count": info["stats"]["domain_count"],
        }

    info = instance_info(instance)

    return _format(info)


def fetch_toots():
    toots = fetch_timeline(instance)
    return pd.DataFrame(toots)


def wordcloud(text):
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import numpy as np

    # def plot_cloud(wordcloud):
    #     plt.figure(figsize=(40, 30))
    #     plt.imshow(wordcloud)
    #     plt.axis("off")

    # circle mask
    width, height = 1000, 1000
    gap = 20
    x, y = np.ogrid[:width, :height]
    mask = (x - width // 2) ** 2 + (y - height // 2) ** 2 > (width // 2 - gap) ** 2
    mask = 255 * mask.astype(int)

    wordcloud = WordCloud(
        width=width,
        height=height,
        random_state=1,
        background_color="white",
        colormap="plasma",
        collocations=False,
        stopwords=STOPWORDS,
        mask=mask,
    ).generate(text)

    return wordcloud.to_image()
    # return plot_cloud(wordcloud)


def get_top_n_words(corpus, n=None):
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(stop_words="english").fit(corpus)
    bow = vec.transform(corpus)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def normalize_text_column(df_column, remove_links=True):
    import re

    HTML_EXP = re.compile("<.*?>")
    df_column = df_column.str.replace(HTML_EXP, "", regex=True)

    if remove_links:
        df_column = df_column.apply(lambda x: re.split("https:\/\/.*", str(x))[0])

    return df_column


def column_to_text(df_column):
    return df_column.str.cat(sep=" ")


st.write(f"# Instance: `{instance}`")

st.write("## Instance stats")
info_dict = fetch_info()
st.write(pd.DataFrame(info_dict.items()))

st.write("## What are people talking about?")
toots = fetch_toots()


toots_timeline = normalize_text_column(toots.content)
text = column_to_text(toots_timeline)
cloud = wordcloud(text)
st.image(cloud)

words = get_top_n_words(toots_timeline, n=20)
df = pd.DataFrame(words, columns=["word", "count"]).set_index("word")
st.write(df)
st.bar_chart(df, y="count")

st.write(
    """
    ## Who is on the instance?
    TODO
    """
)

st.write("## What does the neighborhood of the instance look like?")


# TODO: actually fetch the neighboorhood dynamically for current `instance`.
graph_data = convert_for_web()
size_fn = lambda node: 25 if node["name"] == instance.uri else 10
nodes = [
    Node(id=node["id"], label=node["name"], size=size_fn(node))
    for node in graph_data["nodes"]
]
edges = [
    Edge(source=edge["source_id"], target=edge["target_id"])
    for edge in graph_data["links"]
]

config = Config(
    width="80%",
    height=800,
    # **kwargs
)

agraph(nodes=nodes, edges=edges, config=config)
