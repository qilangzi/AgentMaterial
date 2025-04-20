from sqlalchemy import Column, Integer, String, LargeBinary
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

