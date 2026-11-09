import uvicorn
from fastapi import FastAPI
import string
# 创建一个 FastAPI 应用实例 创建一个fastapi实例
app = FastAPI()
# 定义一个 GET 请求的路由    直接写async定义函数 你不用await就无所谓了 反正都是协程没有启动
@app.get("/", summary="第一个测试")
async def read_root():
    return {"Hello": "World"}
@app.get("/items/{item_id}", summary="获取指定参数")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


# 接收? skip=? & limit = ?
@app.get("/items", summary="分页")
async def read_item(skip: int = 0, limit: int = 10):
    print("222")
    return {"skip": skip, "limit": limit}

if __name__=="__main__":
    uvicorn.run("demofastapi:app", reload=True, host="127.0.0.1", port=8000) #这种就可以断点 命令行断点不了