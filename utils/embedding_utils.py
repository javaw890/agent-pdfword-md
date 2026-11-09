from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from conf.embedding_config import embedding_config


_bge_m3_ef=None
def get_bge_m3_ef():
    #全局单例模式
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef
    model_name = embedding_config.bge_m3
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3EmbeddingFunction(

        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
    return _bge_m3_ef

def generate_embeddings(texts):
    """
    为文本生成向量嵌入 只获取了稠密向量
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)
    processed_sparse = []
    #获取稀疏的算法
    for i in range(len(texts)):
        #根据行里得数据那列所以以及data
        sparse_indices = embeddings["sparse"].indices[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_data = embeddings["sparse"].data[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices,sparse_data)}
        processed_sparse.append(sparse_dict)
    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }