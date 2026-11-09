import os
from dataclasses import dataclass

from dotenv import load_dotenv
#加载配置文件
load_dotenv()
#定义专门配置文件 用来临时用

#常用于配置文件
@dataclass
class MinerUConfig:
    base_url:str
    api_token:str
    api_token_test:str
    base_url_test:str
mineru_config=MinerUConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),
    api_token=os.getenv("MINERU_API_TOKEN",""),
    api_token_test=os.getenv("MINERU_API_TOKEN",""),
    base_url_test=os.getenv("MINERU_BASE_URL","")
)
