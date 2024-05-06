from typing import List
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import SessionLocal
from table_user import User
from table_post import Post
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet

app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db


@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db)):
    result = db.query(User).filter(User.id == id).one_or_none()
    if result:
        return result
    else:
        raise HTTPException(404, "User with id = {%s} not found" % id)


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):
    return db.query(Feed).filter(Feed.user_id == id).order_by(Feed.time.desc()).limit(limit).all()


@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int, db: Session = Depends(get_db)):
    result = db.query(Post).filter(Post.id == id).one_or_none()
    if result:
        return result
    else:
        raise HTTPException(404, "Post with id = {%s} not found" % id)


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post(id: int, limit: int = 10, db: Session = Depends(get_db)):
    return db.query(Feed).filter(Feed.post_id == id).order_by(Feed.time.desc()).limit(limit).all()


@app.get("/post/recommendations/")
def get_recommendations(limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Post.id, Post.text, Post.topic, func.count(Feed.action)). \
        join(Feed). \
        filter(Feed.action == "like"). \
        group_by(Post.id). \
        order_by(func.count(Feed.action).desc()). \
        limit(limit). \
        all()
    return result
