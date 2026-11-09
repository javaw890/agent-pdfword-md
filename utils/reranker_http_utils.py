import dashscope
from http import HTTPStatus

from conf.reranker_config import reranker_config

# 以下为华北2（北京）地域的配置，调用时请将{WorkspaceId}替换为真实的业务空间ID，各地域的配置不同。
dashscope.base_http_api_url = reranker_config.text_rerank_base_url
dashscope.api_key = reranker_config.text_rerank_api_key
def text_rerank(query:str,documents:list[str],instruct:str='',top_n:int=0,model:str=reranker_config.text_rerank_model)->list[str]:
    if not instruct:
        instruct=''
    if not top_n:
        top_n=len(documents)
    response = dashscope.TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        instruct=instruct,
        return_documents=False
    )
    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")


    results = response.output.get("results", [])
    scores = [0.0] * len(documents)
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        scores[int(index)] = float(score)

    return scores