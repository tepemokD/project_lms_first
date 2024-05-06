from example.database import Base
from sqlalchemy import Column, Integer, String


class User(Base):
    __tablename__ = "user"
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    id = Column(Integer, primary_key=True)
    os = Column(String)
    source = Column(String)
