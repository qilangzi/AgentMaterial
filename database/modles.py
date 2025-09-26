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


class MagneticCryst(Base):
    __tablename__ = 'magneticCryst'
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(String)
    material_name = Column(String)
    material_url = Column(String)
    materImageUrl = Column(String)
    parentSpaceGroupUrl = Column(String)
    transitionTemperature = Column(String)
    experimentTemperature = Column(String)
    latticeParameters = Column(String)
    bnsMagneticSpaceGroup = Column(String)
    tableMagneticAtoms = Column(String)
    tableMagneticAtom = Column(String)
    tableNoMagneticAtoms = Column(String)
    tableNoMagneticAtom = Column(String)
    getmirrepsurl = Column(String)
    propagationvector = Column(String)