from contextlib import asynccontextmanager

from fastapi import FastAPI

from database.connection import Base, engine
from database.modles import *
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    print("创建 PostgreSQL 表结构中...")
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL 表已就绪")
    yield