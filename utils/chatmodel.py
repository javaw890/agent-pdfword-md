from langchain_openai import ChatOpenAI
from conf.llm_config import lm_config



import threading




model = None
_lock = threading.Lock()
def get_ChatOpenAI(config=None):
    '''
    创建model实例化
    :return:
    '''
    if config is None:
        config=lm_config
     #要去langchain官方去找不能去官方的去找
    open_ai = ChatOpenAI(model=config.llm_model, api_key=config.api_key, base_url=config.base_url,
                    temperature=config.llm_temperature, extra_body={"enable_thinking": False})

    return open_ai

def get_chat_model():
    global model

    with _lock:  # 加锁
        if model is None:  # 第二次检查，拿到锁后再确认一次
            model = get_ChatOpenAI()
    return model


