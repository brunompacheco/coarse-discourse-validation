# -*- coding: utf-8 -*-
import nltk
import os
import pandas as pd
import math

from collections import Counter
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

nltk.download('stopwords')


def structure_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    ''' Post depth '''
    df['post_depth_normalized'] = df['post_depth'] \
        / df['comments_in_discussion']

    def apply_to_title(body_function):
        def f(row):
            if row['post_depth'] == 0:
                return body_function(row['thread_title'])
            else:
                return 0
        return f
    ''' Number of sentences '''
    def sentences_n(body): return len(nltk.tokenize.sent_tokenize(body))
    df['n_sentences'] = df['body'].map(sentences_n)
    df['n_sentences_title'] = df.apply(apply_to_title(sentences_n), axis=1)

    ''' Number of words '''
    def words_n(body): return len(nltk.tokenize.word_tokenize(body))
    df['n_words'] = df['body'].map(words_n)
    df['n_words_title'] = df.apply(apply_to_title(words_n), axis=1)

    ''' Number of characters '''
    # creates a tokenizer to get words (stripping whitespaces)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # sums the lengths of every word after the tokenizer
    def chars_n(body): return sum(map(len, tokenizer.tokenize(body)))
    df['n_chars'] = df['body'].map(chars_n)
    df['n_chars_title'] = df.apply(apply_to_title(chars_n), axis=1)

    ''' Parents number of sentences, words and characters '''
    df['parent_n_sentences'] = 0
    df['parent_n_words'] = 0
    df['parent_n_chars'] = 0

    for i, row in df.iterrows():
        try:
            parent = df[df['thread_title'] == row['thread_title']][df['id'] == row['in_reply_to']].iloc[0]
            df.at[i, 'parent_n_sentences'] = parent['n_sentences']
            df.at[i, 'parent_n_words'] = parent['n_words']
            df.at[i, 'parent_n_chars'] = parent['n_chars']
        except:
            pass  # if the comment has no parent

    return df


def author_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    # to avoid problems with deleted accounts
    def is_thread_author(row):
        if row['author'] == '[deleted]':
            return False
        else:
            return row['author'] == row['thread_author']

    df['is_thread_author'] = df.apply(is_thread_author, axis=1)

    def is_parent_author(row):
        try:
            parent_author = df[df['id'] == row['in_reply_to']].iloc[0]['author']
            return (row['author'] == parent_author) and (row['author'] != '[deleted]')
        except IndexError:  # in case there is no 'in_reply_to' in the dataset
            return False

    df['is_parent_author'] = df.apply(is_parent_author, axis=1)

    return df


def community_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    df['subreddit'] = df['subreddit'].astype('category')

    return df


''' TF-IDF weighting functions '''


def docs_containing(word, docs):
    def f(doc): return doc.count(word) > 0
    return list(filte(f, docs))


def tf(word, doc):
    return doc.count(word) / float(len(doc))


def idf(word, docs):
    return math.log( len(docs) / float(len(docs_containing(word, docs))) )


def weighted_vectorizer(token_list, token_dict):
    feature = []

    for token in token_list:
        try:
            feature.append(token_dict[token])
        except KeyError:
            feature.append(0)

    return feature


def content_punctuation_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    ''' Body ngrams '''
    df['tokens'] = df['body'].apply(lambda body: nltk.word_tokenize(body))
    df['bigrams'] = df['tokens'].apply(lambda tokens: list(nltk.bigrams(tokens)))
    df['trigrams'] = df['tokens'].apply(lambda tokens: list(nltk.trigrams(tokens)))

    unigrams_frequency = df['tokens'].apply(Counter).sum()
    bigrams_frequency = df['bigrams'].apply(Counter).sum()
    trigrams_frequency = df['trigrams'].apply(Counter).sum()

    df['TF-IDF'] = [dict() for _ in range(len(df))]

    for i, row in df.iterrows():
        # TODO: Make feature vector already as columns from here
        for token in row['tokens']:
            row['TF-IDF'][token] = tf(token, row['tokens']) * idf(token, df['tokens'])

        try:
            for token in row['bigrams']:
                row['TF-IDF'][token] = tf(token, row['bigrams']) * idf(token, df['bigrams'])
        except ZeroDivisionError:
            print(token)
            raise

        for token in row['trigrams']:
            row['TF-IDF'][token] = tf(token, row['trigrams']) * idf(token, df['trigrams'])

    df['unigrams_feature'] = None
    df['bigrams_feature'] = None
    df['trigrams_feature'] = None

    top_unigrams = [tk for tk, freq in unigrams_frequency.most_common() if freq >= 50]
    top_bigrams = [tk for tk, freq in bigrams_frequency.most_common() if freq >= 50]
    top_trigrams = [tk for tk, freq in trigrams_frequency.most_common() if freq >= 50]

    for i, row in df.iterrows():
        df.at[i, 'unigrams_feature'] = weighted_vectorizer(top_unigrams, row['TF-IDF'])
        df.at[i, 'bigrams_feature'] = weighted_vectorizer(top_bigrams, row['TF-IDF'])
        df.at[i, 'trigrams_feature'] = weighted_vectorizer(top_trigrams, row['TF-IDF'])

    ''' Title ngrams '''    
    df['title_tokens'] = df['thread_title'].apply(lambda title: nltk.word_tokenize(title))
    df['title_bigrams'] = df['title_tokens'].apply(lambda tokens : list(nltk.bigrams(tokens)))
    df['title_trigrams'] = df['title_tokens'].apply(lambda tokens : list(nltk.trigrams(tokens)))

    # gets unique values, excluding comments without title
    titles_tokens = pd.Series(list(set(df['thread_title']))).apply(lambda title: nltk.word_tokenize(title))
    titles_bigrams = titles_tokens.apply(lambda tokens : list(nltk.bigrams(tokens)))
    titles_trigrams = titles_tokens.apply(lambda tokens : list(nltk.trigrams(tokens)))

    unigrams_frequency = titles_tokens.apply(Counter).sum()
    bigrams_frequency = titles_bigrams.apply(Counter).sum()
    trigrams_frequency = titles_trigrams.apply(Counter).sum()

    df['title_TF-IDF'] = [dict() for _ in range(len(df))]

    for i, row in df.iterrows():
        # TODO: Make feature vector already as columns from here
        for token in row['title_tokens']:
            row['title_TF-IDF'][token] = tf(token, row['title_tokens']) \
                                         * idf(token, df['title_tokens'])

        try:
            for token in row['title_bigrams']:
                row['title_TF-IDF'][token] = tf(token, row['title_bigrams']) \
                                             * idf(token, df['title_bigrams'])
        except ZeroDivisionError:
            print(token)
            raise

        for token in row['title_trigrams']:
            row['title_TF-IDF'][token] = tf(token, row['title_trigrams']) \
                                         * idf(token, df['title_trigrams'])

    df['title_unigrams_feature'] = None
    df['title_bigrams_feature'] = None
    df['title_trigrams_feature'] = None

    top_unigrams = [tk for tk, freq in unigrams_frequency.most_common() if freq >= 50]
    top_bigrams = [tk for tk, freq in bigrams_frequency.most_common() if freq >= 50]
    top_trigrams = [tk for tk, freq in trigrams_frequency.most_common() if freq >= 50]

    for i, row in df.iterrows():
        df.at[i, 'title_unigrams_feature'] = weighted_vectorizer(
            top_unigrams,
            row['title_TF-IDF'])
        df.at[i, 'title_bigrams_feature'] = weighted_vectorizer(
            top_bigrams,
            row['title_TF-IDF'])
        df.at[i, 'title_trigrams_feature'] = weighted_vectorizer(
            top_trigrams,
            row['title_TF-IDF'])

    # drops columns created for development
    df.drop(coolumns=['tokens', 'bigrams', 'trigrams', 'TF-IDF', 'title_tokens', 'title_bigrams', 'title_trigrams', 'title_TF-IDF'], inplace=True)

    return df


def main(proj_root):
    df_path = os.path.join(proj_root, 'data', 'interim', 'clean_data.pkl')

    print("Reading dataset", end=' ')
    df = pd.read_pickle(df_path)
    print("- Done!")

    print("Making structure features", end=' ')
    df = structure_features(df)
    print("- Done!")

    print("Making author features", end=' ')
    df = author_features(df)
    print("- Done!")

    print("Making community features", end=' ')
    df = community_features(df)
    print("- Done!")

    print("Making content+punctuation features", end=' ')
    df = content_punctuation_features(df)
    print("- Done!")

    # renames a couple of columns to improve readability
    df['n_branches_in_discussion'] = df['branches_number']
    df['n_comments_in_discussion'] = df['comments_in_discussion']

    # drops useless columns
    # df.drop(columns=[
    #     'annotations',
    #     'author',
    #     'body',
    #     'branches_number',
    #     'comments_in_discussion',
    #     'comments_number',
    #     'id',
    #     'in_reply_to',
    #     'thread_author',
    #     'thread_title',
    #     'url',
    #     ], inplace=True)
    # WILL LEAVE THIS STEP FOR THE ALGORITHMS

    print("Saving dataset", end=' ')
    df.to_pickle(os.path.join(proj_root, 'data', 'processed',
                              'processed_dataset.pkl'))
    print("- Done!")


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)