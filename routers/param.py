import fastapi
from fastapi import APIRouter,Depends
from database.connection import *
from database.modles import *
from pydantic import BaseModel
from sqlalchemy.orm import Session
router = APIRouter(
    prefix='/material',
    tags=['material'],
)

class TextInput(BaseModel):
    text:str|None=None
    filefrom:str|None=None
    fileDIO:str|None=None
    status:int|None=None
    # embedding:bytes|None=None
@router.post('/test')
#测试数据库
async def test(textinpput:TextInput, db:Session=Depends(get_db)):
            db.text=Text(text=textinpput.text,
                     filefrom=textinpput.filefrom,
                     fileDIO=textinpput.fileDIO,
                     status=textinpput.status)
                     # embedding=textinpput.embedding)
            db.add(db.text)
            db.commit()
            db.refresh(db.text)
            return {'message': 'Text stored successful'}


@router.get("/texts/{text_id}", response_model=TextInput)
async def read_text(text_id: int, db: Session = Depends(get_db)):
            db_text = db.query(Text).filter(Text.id == text_id).first()
            return db_text

# 获取所有 Texts
@router.post("/texts/", response_model=list[TextInput])
async def read_texts(skip: int = 0, limit: int = 1, db: Session = Depends(get_db)):
        texts = db.query(Text).offset(skip).limit(limit).all()
        return texts



@router.put("/texts/{text_id}", response_model=TextInput)
async def update_text_route(text_id: int, text_put: TextInput, db: Session = Depends(get_db)):
        db_text = db.query(Text).filter(Text.id == text_id).first()
        print(db_text)
        if db_text:
            db_text.text = text_put.text
            db_text.filefrom =text_put.filefrom
            db_text.fileDIO =text_put.fileDIO
            db_text.status =text_put.status
            db.commit()
            db.refresh(db_text)
        return db_text

# 删除 Text
@router.delete("/texts/{text_id}")
async def delete_text_route(text_id: int, db: Session = Depends(get_db)):
        db_text  = db.query(Text).filter(Text.id == text_id).first()
        if db_text:
            db.delete(db_text)
            db.commit()
        return db_text


@router.post('/add')
async def add(x:int, y:int):
    return {'result': x + y}