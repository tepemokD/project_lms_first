import pandas as pd
from sqlalchemy import create_engine


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


def load_features():
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    df_feed = batch_load_sql("""
    SELECT DISTINCT post_id, user_id 
    FROM public.feed_data 
    """)
    # df_feed = batch_load_sql("""
    # SELECT DISTINCT post_id, user_id
    # FROM public.feed_data
    # WHERE action = 'like'
    # """)

    df_user = pd.read_sql('SELECT * FROM public.user_data', engine)
    # df_text = batch_load_sql('SELECT * FROM public.perepelitsa_text_lesson_22')
    df_text = pd.read_sql('SELECT * FROM public.perepelitsa_text_lesson_22', engine)
    df = pd.merge(df_feed,
                  df_text,
                  on='post_id',
                  how='left')

    df = pd.merge(df,
                  df_user,
                  on='user_id',
                  how='left')

    # df['month'] = df['timestamp'].dt.month
    # df['hour'] = df['timestamp'].dt.hour

    # df.drop(['action', 'text'],
    #         axis=1,
    #         inplace=True)

    # df = df.set_index(['user_id', 'post_id'])

    return df


# df_user = pd.read_csv('user_data.csv')
# df_feed = pd.read_csv('feed_data.csv')
# df_text = pd.read_csv('post_text_df_data.csv')

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

# записываем таблицу
# df_user.to_sql('perepelitsa_user_lesson_22',не надо
#                con=engine,
#                schema='public',
#                if_exists='replace')
#
# df_feed.to_sql('perepelitsa_feed_lesson_22', не надо
#                con=engine,
#                schema='public',
#                if_exists='replace')

# df_text.to_sql('perepelitsa_text_lesson_22',
#                con=engine,
#                schema='public',
#                if_exists='replace')


print(load_features().head())
