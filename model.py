import string

import pandas as pd

pd.set_option('display.max_columns', 100)

from catboost import CatBoostClassifier, Pool

import spacy
#from nltk.stem import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

mlp.rcParams['lines.linewidth'] = 5

mlp.rcParams['xtick.major.size'] = 10
mlp.rcParams['xtick.major.width'] = 5
mlp.rcParams['xtick.labelsize'] = 10
mlp.rcParams['xtick.color'] = '#FF5533'

mlp.rcParams['ytick.major.size'] = 10
mlp.rcParams['ytick.major.width'] = 5
mlp.rcParams['ytick.labelsize'] = 10
mlp.rcParams['ytick.color'] = '#FF5533'

mlp.rcParams['axes.labelsize'] = 10
mlp.rcParams['axes.titlesize'] = 10
mlp.rcParams['axes.titlecolor'] = '#00B050'
mlp.rcParams['axes.labelcolor'] = '#00B050'


def download():
    '''
    Загрузка таблиц из БД
    :return:
    датафреймы трех таблиц
    '''
    url = 'postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml'
    user_data = pd.read_sql('SELECT * FROM public.user_data',
                            url)
    post_text_df = pd.read_sql('SELECT * FROM public.post_text_df',
                               url)
    feed_data = pd.read_sql('SELECT * FROM public.feed_data LIMIT 10000000',
                            url)
    return user_data, post_text_df, feed_data


def save_database(data: pd.DataFrame, fname: str):
    '''
    Сохранение датафрейма в файл
    :param data:
        Датафрейм
    :param fname:
        Название файла с расширением
    :return:
    '''
    data.to_csv(fname, index=False)


def save_model_catboost(model: CatBoostClassifier, fname: str):
    '''
        Запись модели CatBoostClassifier в файл
    :param model:
        Модель
    :param fname:
        Название файла
    :return:
        None
    '''
    model.save_model(fname,
                     format="cbm")


nlp = spacy.load('en_core_web_sm')


def preprocessing_text(line: str, tok=nlp) -> str:
    line = line.lower()
    line = re.sub(r'[{}]'.format(string.punctuation), " ", line)
    line = line.replace('\n\n', ' ').replace('\n', ' ')
    line = re.sub(r'[{}]'.format(string.digits), " ", line)

    line = nlp(line)
    line = ' '.join([token.lemma_.lower() for token in line])
    return line


if __name__ == '__main__':
    # _, _, feed_data = download()
    # save_database(feed_data, 'feed_data.csv')

    user_data = pd.read_csv('user_data.csv')
    feed_data = pd.read_csv('feed_data.csv', parse_dates=['timestamp'])
    '''
    # post_text_df_data = pd.read_csv('post_text_df.csv')
    '''

    # Работа с параметрами/фичами
    feed_data = feed_data[feed_data['action'] != 'like']

    '''
    # Построение матрицы категориальных признаков для текстов постов с момощью лемматизации
    # tfidf = TfidfVectorizer(preprocessor=preprocessing_text,
    #                         stop_words='english')
    #
    # tfidf_data = pd.DataFrame(tfidf.fit_transform(post_text_df_data['text']).toarray(),
    #                           index=post_text_df_data['post_id'],
    #                           columns=tfidf.get_feature_names_out())
    #
    # save_database(tfidf_data, 'post_text_df_tfidf.csv')

    # Уменьшение количества признаков для текста через PCA и dbscan до 1го
    # tfidf_data = pd.read_csv('post_text_df_tfidf.csv')
    #
    # tfidf_data = tfidf_data - tfidf_data.mean()
    #
    # pca = PCA(n_components=13)
    # pca_decomp = pd.DataFrame(pca.fit_transform(tfidf_data))
    #
    #
    # save_database(pca_decomp, 'pca_decomp.csv')
    pca_decomp = pd.read_csv('pca_decomp.csv')
    '''
    # Подбор параметров dbscan
    '''eps_list = [0.05, 0.07]
    min_samples_list = [3, 5, 7]
    df = pd.DataFrame({'labels': []})

    for eps in eps_list:
        for min_samples in min_samples_list:
            dbscan = DBSCAN(eps=eps,
                            min_samples=min_samples)
            dbscan.fit(pca_decomp)

            df_ = pd.Series(dbscan.labels_)
            df_ = df_.value_counts()
            df_ = pd.DataFrame({'labels': df_.index, f'e={eps} s={min_samples}': df_.values})

            df = df.merge(df_, how='outer', on='labels')

    df = pd.melt(df, id_vars='labels', var_name='param_model', value_name='count')

    sns.barplot(data=df, x='labels', y='count', hue='param_model')
    plt.show()'''
    '''
    # dbscan = DBSCAN(eps=0.05,
    #                 min_samples=7)
    # dbscan.fit(pca_decomp)
    # post_text_df_data['Labels'] = dbscan.labels_

    # save_database(post_text_df_data, 'post_text_df_data.csv')
    '''
    post_text_df_data = pd.read_csv('post_text_df_data.csv')

    # Объединение таблиц
    df = pd.merge(feed_data,
                  post_text_df_data,
                  on='post_id',
                  how='left')

    df = pd.merge(df,
                  user_data,
                  on='user_id',
                  how='left')

    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    df.drop(['action', 'text'],
            axis=1,
            inplace=True)

    df = df.set_index(['user_id', 'post_id'])

    '''
    # df.to_csv('df_model.csv')
    '''

    # Выделение train и test по timestamp
    border_date = '2021-12-10'
    df_train = df[df['timestamp'] < border_date]
    df_test = df[df['timestamp'] >= border_date]

    df_train = df_train.drop(['timestamp'],
                             axis=1)
    df_test = df_test.drop(['timestamp'],
                           axis=1)

    X_train = df_train.drop(['target'],
                            axis=1)
    X_test = df_test.drop(['target'],
                          axis=1)

    y_train = df_train['target']
    y_test = df_test['target']

    # Обучение модели и подбор гиперпараметров
    cat_obj = ['topic', 'Labels', 'gender', 'country', 'city', 'exp_group', 'os', 'source',
               'month', 'hour']

    model_cbc = CatBoostClassifier(iterations=100,
                                   learning_rate=0.2,
                                   depth=9)

    model_cbc.fit(X_train, y_train, cat_obj)

    print('Качество при обучении', roc_auc_score(y_train, model_cbc.predict_proba(X_train)[:, 1]))
    print('Качество на тесте', roc_auc_score(y_test, model_cbc.predict_proba(X_test)[:, 1]))

    sns.barplot(x=model_cbc.feature_importances_, y=X_train.columns)
    plt.show()
    '''
    # save_model_catboost(model_cbc, 'catboost_me')
    '''