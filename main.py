#main file
import streamlit as st
import time
import requests
import math
import pandas as pd
from wordcloud import WordCloud
import nltk
import string
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from requests.exceptions import SSLError
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())

def plot_top_words(model, feature_names, n_top_words, title, n_components):
    fig, axes = plt.subplots(math.ceil(n_components/5), 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    st.pyplot(fig)
    
def get_request(url,parameters=None, steamspy=False):
    """Return json-formatted response of a get request using optional parameters.
    
    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request
    
    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)
        
        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' '*10)
        
        # recursively try again
        return get_request(url, parameters, steamspy)
    
    if response:
        return response.json()
    else:
        # We do not know how many pages steamspy has... and it seems to work well, so we will use no response to stop.
        if steamspy:
            return "stop"
        else :
            # response is none usually means too many requests. Wait and try again 
            print('No response, waiting 10 seconds...')
            time.sleep(10)
            print('Retrying.')
            return get_request(url, parameters, steamspy)
        
def plot_nmf_topics(data_samples, n_features, stop_words, n_components, n_top_words, init, title):
    """Plot NMF topics."""
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words=stop_words
    )
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    
    nmf = NMF(
    n_components=n_components,
    random_state=1,
    init=init,
    beta_loss="frobenius",
    alpha_W=0.00015,
    alpha_H=0.00015,
    l1_ratio=1,
    ).fit(tfidf)
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(
        nmf, tfidf_feature_names, n_top_words, title, n_components
    )
        
@st.cache_data
def get_steam_df():
    """Return a list of all steam games.
    
    Returns
    -------
    
        list of all steam games
    """
    return pd.DataFrame(get_request("https://api.steampowered.com/ISteamApps/GetAppList/v2/?")["applist"]["apps"]).set_index("appid")

@st.cache_data
def parse_steamreviews_request(appid):
    """Parser to handle SteamSpy API data."""
    num_per_page = 100
    max_good_review = 300  # max number of good reviews to return
    max_bad_review = 100
    good_review_count = 0
    bad_review_count = 0
    good_review_list = []
    bad_review_list = []
    url = "https://store.steampowered.com/appreviews/" + str(appid)
    print(url)
    parameters = {"json": 1, "cursor": "*", "num_per_page": num_per_page, "language": "english", "purchase_type": "all", "review_type": "positive"}
    #see cursor
    #https://partner.steamgames.com/doc/store/getreviews
    json_data = get_request(url, parameters)
    summary = json_data['query_summary']
    wnl = WordNetLemmatizer()
    while good_review_count < max_good_review or bad_review_count < max_bad_review:
        # if we have not reached the maximum number of good or bad reviews, and there are still reviews to fetch
        if summary["num_reviews"] == 0:
            break
        # if we have not reached the maximum number of good reviews, and there are still good reviews to fetch
        if good_review_count < max_good_review:
            json_data = get_request(url, parameters)
            for review in json_data["reviews"]:
                good_review_count += 1
                lemmatized_string = ' '.join([wnl.lemmatize(words) for words in nltk.word_tokenize(review["review"])])
                good_review_list.append(lemmatized_string)
        # if we have not reached the maximum number of bad reviews, and there are still bad reviews to fetch
        elif bad_review_count < max_bad_review:
            parameters["review_type"] = "negative"
            if bad_review_count == 0:
                # reset the cursor to the beginning for bad reviews
                parameters["cursor"] = "*"
            json_data = get_request(url, parameters)
            for review in json_data["reviews"]:
                bad_review_count += 1
                bad_review_list.append(review["review"])
        # get next page of reviews
        parameters["cursor"] = json_data["cursor"]
        summary = json_data['query_summary']
        #st.write(json_data)
    return good_review_list, bad_review_list, summary

st.write("Search for a Steam Game")
search_input = st.text_input("Search Steam Game", key="search_input")
if search_input == "":
    search_request = False
else:
    search_request = True
st.write(search_input)
df = pd.DataFrame(get_steam_df())
df = df[df["name"].str.contains(search_input, case=False, na=False)]
if search_request:
    appname = st.selectbox("Select game", df["name"], disabled=not search_request)
    appid = df[df["name"]==appname].index[0]
    extra_stop_words = {"lot","10","h1","n't", "game", "games", "play", "steam", "valve", "played", "playing"}
    extra_stop_words = extra_stop_words.union(set(appname.lower().split()))
    stop_words = stopwords.union(extra_stop_words)
    good_review_list, bad_review_list, summary = parse_steamreviews_request(appid)
    n_samples = 1000
    n_features = 400
    n_components = 20
    n_top_words = 3
    batch_size = 128
    init = "nndsvda"
    
    st.write("Good Reviews:")
    good_wordcloud = WordCloud(width=800,stopwords=stop_words, height=400, background_color='black').generate(' '.join(good_review_list))   
    st.image(good_wordcloud.to_array())
    plot_nmf_topics(good_review_list, n_features, list(stop_words), n_components, n_top_words, init, title="Topics in Good Reviews")
    
    st.write("Bad Reviews:")
    bad_wordcloud = WordCloud(width=800,stopwords=stop_words, height=400, background_color='black').generate(' '.join(bad_review_list))   
    st.image(bad_wordcloud.to_array())
    plot_nmf_topics(bad_review_list, n_features, list(stop_words), n_components, n_top_words, init, title="Topics in Bad Reviews")  

    

