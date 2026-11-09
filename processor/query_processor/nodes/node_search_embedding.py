from conf.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.my_logging import logger
from processor.query_processor.state import QueryGraphState
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import format_json
from utils.milvus_utils import get_milvu_sclient
from utils.minio_utils import create_hybrid_search_requests, hybrid_search


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"
     def process(self, state: QueryGraphState) -> QueryGraphState:
         """
         核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
         流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
         :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                       {
                           "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                           "item_names": list[str],  # step7已确认的标准化商品名列表
                       }

         :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                  {
                      "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                      # 每个元素为一条匹配的向量数据，含业务字段
                  }
         """

         try:

             # 1、用户问题和已确认商品名
             query = state.get("rewritten_query")
             item_names = state.get("item_names")
             if not query:
                 logger.error("用户问题没有")
             if not item_names:
                 logger.error("并没有已确认的商品名列表")
             embeddings = generate_embeddings([query])
             dense_ = embeddings.get("dense")[0]
             sparse_ = embeddings.get("sparse")[0]
             # 4、处理 item_names 中的引号，防止注入或语法错误
             expr = None
             if item_names:
                 # quoted = ", ".join(f'"{v}"' for v in item_names)
                 # expr = f"item_name in [{quoted}]"
                 # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
                 expr = f'item_name in {item_names}'
                 logger.info(f"过滤条件: {expr}")
             else:
                logger.info("未指定商品名过滤，将全库检索")
             reqs = create_hybrid_search_requests(dense_vector=dense_, sparse_vector=sparse_, expr=expr,
                 limit=10)
             client = get_milvu_sclient()
             collection_name = milvus_config.chunks_collection
             res = hybrid_search(
                 client=client,
                 collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
                 reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
                 ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比，各占50%（可按业务调优）
                 output_fields=["chunk_id", "content", "item_name"]  # 指定返回的业务字段
             )
             logger.info("开始执行 Milvus 混合检索...")
             # 7、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
             return {"embedding_chunks": res[0] if res else []}


         except Exception as e:
             #一旦有问题直接降级处理 否则你就是抛异常
             logger.exception(f"向量搜索失败: {e}")
             return {}





if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(format_json(result, indent=4))