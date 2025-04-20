from sqlalchemy import Column, Integer, String, LargeBinary
import sqlalchemy
from sqlalchemy.orm import declarative_base
Base = declarative_base()

class Text(Base):
    __tablename__ = 'texts'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    # embedding = Column(LargeBinary, nullable=False)
    status=Column(Integer)
    fileDIO = Column(String)
    filefrom = Column(String)

class Electromagentic(Base):
    __tablename__ = 'electromagenticExperiment'
    id = Column(Integer, primary_key=True, index=True)
    pathname = Column(String)
    text = Column(sqlalchemy.Text)
    dio = Column(String)
    keywords = Column(String)
    principle = Column(String)
    materials = Column(String)
    methods = Column(String)
    measure = Column(String)
    conclusion = Column(String)
    materials_embedding = Column(String)
    principle_embedding = Column(String)
    methods_embedding = Column(String)
    measure_embedding = Column(String)
    conclusion_embedding = Column(String)
    status=Column(Integer, default=1)
