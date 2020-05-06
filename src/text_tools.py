import pandas as pd
import numpy as np
from datetime import datetime
from NLPInfrastructure.normalizer import SentenceNormalizer

from NLPInfrastructure.resources import stopWords, prepositions, postWords, Not1Gram, stophashtags


class TextTools:
    def create_graph(self, messages_df: pd.DataFrame):
        messages_df['clean_text'] = self._normalize_texts(messages_df['text'])
        allkeywords = self._text_to_allkeywords(messages_df)

        nodes = allkeywords.groupby(['message_id'], as_index=False)['word'].count()
        nodes.columns = ['id', 'weight']

        # remove words in just one message
        keywords_dic = allkeywords.groupby(['word'], as_index=False)['message_id'].count()
        keywords_dic = keywords_dic[keywords_dic['message_id'] > 1]

        # create keywords dictionary
        keywords_dic.reset_index(drop=True, inplace=True)
        keywords_dic.reset_index(inplace=True)
        keywords_dic['index'] = keywords_dic['index'].astype(int)
        keywords_dic.drop(['message_id'], axis=1, inplace=True)

        allkeywords = allkeywords[allkeywords['word'].isin(keywords_dic['word'])]
        allkeywords = allkeywords.merge(keywords_dic, how='left', left_on='word', right_on='word')
        allkeywords.drop(allkeywords.columns.difference(['message_id', 'index']), axis=1, inplace=True)

        edges = allkeywords.merge(allkeywords, how='inner', on='index')
        edges = edges.groupby(['message_id_x', 'message_id_y'], as_index=False)['index'].count()

        edges.columns = ['source', 'target', 'weight']
        # undirected graph
        edges.drop(edges.loc[edges['source'] > edges['target']].index.tolist(), inplace=True)

        # remove selfloops
        edges.drop(edges.loc[edges['source'] == edges['target']].index.tolist(), inplace=True)

        edges.reset_index(drop=True, inplace=True)

        return edges, nodes

    def _text_to_allkeywords(self, data: pd.DataFrame) -> pd.DataFrame:
        data['transactions'] = data['clean_text'].apply(lambda t: list(filter(None, t.split(' '))))

        allkeywords = pd.DataFrame({'message_id': np.repeat(data['id'].values, data['transactions'].str.len()),
                                    'word': np.concatenate(data['transactions'].values)})
        allkeywords['count'] = 1
        allkeywords = allkeywords.groupby(['message_id', 'word'], as_index=False).sum()

        return allkeywords

    def _remove_stopword(self, text):
        text = text.replace('.', ' ').replace(',', ' ')
        words = text.split(' ')
        words_filtered = []
        for w in words:
            if (w not in stopWords) and (w not in prepositions) and (w not in postWords) and (w not in Not1Gram) and (
                    w not in stophashtags) and (not w.isdigit()):
                words_filtered.append(w)

        res = ' '.join(words_filtered)
        res = res.strip()
        return res

    def _normalize_text_atomic(self, text):
        normalizer = SentenceNormalizer()
        text = normalizer.organize_text(text)
        text = normalizer.replace_urls(text, '')
        text = normalizer.replace_emails(text, '')
        text = normalizer.replace_usernames(text, '')
        # text = normalizer.replace_hashtags(text, 'MyHashtag')
        text = normalizer.edit_arabic_letters(text)
        text = normalizer.replace_phone_numbers(text, '')
        text = normalizer.replace_emoji(text, '')
        text = normalizer.replace_duplicate_punctuation(text)

        text = normalizer.replace_consecutive_spaces(text)
        text = self._remove_stopword(text)

        return text

    def _normalize_texts(self, texts):
        return [self._normalize_text_atomic(text) for text in texts]
