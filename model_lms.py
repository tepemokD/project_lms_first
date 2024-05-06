import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from example.schema import PostGet
from example.app import app
from datetime import datetime
from typing import List


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_models():
    model_path = get_model_path("/my/super/path")
    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path)  # пример как можно загружать модели
    return model


def load_features():
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    df_feed = batch_load_sql("""
    SELECT DISTINCT post_id, user_id 
    FROM public.feed_data 
    """)

    df_user = pd.read_sql('SELECT * FROM public.user_data', engine)
    df_text = pd.read_sql('SELECT * FROM public.perepelitsa_text_lesson_22', engine)
    df = pd.merge(df_feed,
                  df_text,
                  on='post_id',
                  how='left')

    df = pd.merge(df,
                  df_user,
                  on='user_id',
                  how='left')

    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    df.drop(['action', 'text'],
            axis=1,
            inplace=True)

    df = df.set_index(['user_id', 'post_id'])

    return df


model = load_models()
features = load_features()


def get_recommended_feed(id: int, time: datetime, limit: int):
    user_features = features.loc[features.user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_features = features.drop(['index', 'text'], axis=1)

    content = features[['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    predict = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predict'] = predict

    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


@app.get("/post/recommendations/", response_class=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return recommended_posts(id, time, limit)
