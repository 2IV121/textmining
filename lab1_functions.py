# -*- coding: utf-8 -*-
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
    )
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from textmining import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#donald_tweets = pd.read_csv(
#    "DonaldTweets.tsv",
#    encoding="utf-8", sep="\t", index_col="Tweet_Id"
#)

donald_tweets = pd.read_csv("DonaldTweets.csv")

sample_corpus = pd.Series([
    "The loveliest of lovely meetings at Gothenburg Book Fair w @KinsellaSophie Thank you 💗 #bokmässan #bookblogger @Marimekkoglobal @wsoykirjat",
    "The 1st day of 3 at #Bokmässan today 📖",
    "This storybook app is a great conversation starter for #parents & #teachers 👨‍👩‍👧‍👦https://youtu.be/LUSz-7dmyRs  FREE DOWNLOAD 🙌🏻 #bokmässan #SEL",
    "1) this is my dream workspace 2) this #bokmässan session on picture books reminds me of my (currently dormant 😭) dream of being a librarian.",
    "The unboxing moment 😇😀😊☺️📚📚📚📙📖📑📓📔📕📖🎆🎇🎈🎂🎊🎉🎁 #mittulricehamn #författarlivet #bokmässan… https://www.instagram.com/p/BZeIyhvDoGP/"
])

my_stop_words_sw = ["och", "det", "att", "i", "en", "jag", "hon",
                "som", "han", "paa", "den", "med", "var", "sig",
                "foer", "saa", "till", "aer", "men", "ett",
                "om", "hade", "de", "av", "icke", "mig", "du",
                "henne", "daa", "sin", "nu", "har", "inte",
                "hans", "honom", "skulle", "hennes", "daer",
                "min", "man", "ej", "vid", "kunde", "naagot",
                "fraan", "ut", "naer", "efter", "upp", "vi",
                "dem", "vara", "vad", "oever", "aen", "dig",
                "kan", "sina", "haer", "ha", "mot", "alla",
                "under", "naagon", "eller", "allt", "mycket",
                "sedan", "ju", "denna", "sjaelv", "detta",
                "aat", "utan", "varit", "hur", "ingen", "mitt",
                "ni", "bli", "blev", "oss", "din", "dessa",
                "naagra", "deras", "blir", "mina", "samma",
                "vilken", "er", "saadan", "vaar", "blivit",
                "dess", "inom", "mellan", "saadant", "varfoer",
                "varje", "vilka", "ditt", "vem", "vilket",
                "sitta", "saadana", "vart", "dina", "vars",
                "vaart", "vaara", "ert", "era", "vilka"]


my_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def first_15_tweets():
    """
    for i, item in enumerate(donald_tweets.Tweet_Text.head()):
        print("Tweet {}: {}".format(i, item))
    """
    global donald_tweets
    for i, item in enumerate(donald_tweets.Tweet_Text.head(15)):
        print("Tweet {}: {}".format(i, item))

def view_table():
    global donald_tweets
    return donald_tweets

def view_table_info():
    global donald_tweets
    return donald_tweets.info()

def filter_emojis():
    global sample_corpus
    def filter_emojis(filter_emojis):
        if filter_emojis:
            encode2ascii = lambda x: x.encode('ascii', errors='ignore').decode('utf-8')
            clean_tweets = sample_corpus.apply(encode2ascii)
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(filter_emojis, filter_emojis=False)

def filter_urls():
    global sample_corpus
    def filter_urls(filter_urls):
        if filter_urls:
            clean_tweets = sample_corpus.str.replace(r'http\S+', '')
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(filter_urls, filter_urls=False)

def create_term_document_matrix(sample_size, min_df=1):
    corpus = donald_tweets.head(sample_size).Tweet_Text.values.astype('U')
    cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
    tfmatrix = cvec.fit_transform(corpus)
    return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())

def create_term_document_matrix3(sample_size, min_df=1):
    corpus = donald_tweets.head(sample_size).Tweet_Text.values.astype('U')
    cvec = CountVectorizer(min_df=min_df, stop_words=None, lowercase=False)
    tfmatrix = cvec.fit_transform(corpus)
    return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())

def make_tdm():
    global donald_tweets
    interact(create_term_document_matrix3, sample_size=widgets.IntSlider(min=1,max=15,step=1,value=3), min_df=fixed(1))

def top_words(num_word_instances, top_words):
    tweets = donald_tweets.Tweet_Text
    tdm_df = create_term_document_matrix3(len(tweets), min_df=2)
    word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
    sorted_words = word_frequencies.sort_values(ascending=False)
    top_sorted_words = sorted_words[:num_word_instances]
    top_sorted_words[:top_words].plot.bar()
    return top_sorted_words

def plot_top_words():
    global donald_tweets
    interact(top_words, num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False), top_words=widgets.IntSlider(min=1, value=30, continuous_update=False))

def make_lowercase():
    global sample_corpus
    def lowercase(lowercase):
        if lowercase:
            clean_tweets = sample_corpus.str.lower()
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(lowercase, lowercase=False)

def remove_small_words():
    global sample_corpus
    def small_words(small_words):
        if small_words:
            clean_tweets = sample_corpus.str.findall('\w{3,}').str.join(' ')
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(small_words, small_words=False)


# remove_stopwords = lambda x: print(x)
remove_stopwords = lambda x: ' '.join(y for y in x.split() if y not in my_stop_words)

def remove_stop_words():
    global donald_tweets
    sample_corpus = donald_tweets.sample(10).Tweet_Text
    def stop_words(stop_words):
        if stop_words:
            clean_tweets = sample_corpus.apply(remove_stopwords)
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(stop_words, stop_words=False)

def plot_top_words_with_filters():
    global donald_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords,  lowercase=False)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def create_term_document_matrix4(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=None, lowercase=False)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, top_words, stop_words, small_words, lower):
        tweets = donald_tweets.Tweet_Text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        tdm_df = create_term_document_matrix4(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        top_sorted_words[:top_words].plot.bar()
        return top_sorted_words
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        top_words=widgets.IntSlider(min=1, value=30, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False)
        )

def plot_top_words_with_custom_stopwords():
    global donald_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def create_term_document_matrix4(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=None, lowercase=False)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, top_words, stop_words, small_words, lower, more_stop_words):
        tweets = donald_tweets.Tweet_Text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        if len(more_stop_words) > 0:
            remove_more_stopwords = lambda x: ' '.join(y for y in x.split() if y not in (x.strip() for x in more_stop_words.split(',')))
            tweets = tweets.apply(remove_more_stopwords)
        tdm_df = create_term_document_matrix4(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        top_sorted_words[:top_words].plot.bar()
        return top_sorted_words
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        top_words=widgets.IntSlider(min=1, value=30, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False),
        more_stop_words=widgets.Text(value='one,', description='Additional stop words:', continuous_update=False))

def plot_wordcloud():
    global donald_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def create_term_document_matrix4(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=None, lowercase=False)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, stop_words, small_words, lower, more_stop_words):
        tweets = donald_tweets.Tweet_Text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        if len(more_stop_words) > 0:
            remove_more_stopwords = lambda x: ' '.join(y for y in x.split() if y not in (x.strip() for x in more_stop_words.split(',')))
            tweets = tweets.apply(remove_more_stopwords)
        tdm_df = create_term_document_matrix4(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        wordcloud = WordCloud(max_font_size=40)
        wordcloud.fit_words(top_sorted_words.to_dict())
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False),
        more_stop_words=widgets.Text(value='one,', description='Additional stop words:', continuous_update=False))

print("Human-Computer Interaction: Social Media Text mining initialized... OK!")
