import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from utils.embedding_utils import get_bge_m3_ef

bgem3 = get_bge_m3_ef()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

docs_embeddings = bgem3.encode_documents(docs)

# 稠密向量：直接 list 化
dense_ = [emb.tolist() for emb in docs_embeddings["dense"]]

# 稀疏向量：逐行从 CSR 里拆成 {维度id: 权重}
sparse = []
sparse_obj = docs_embeddings.get("sparse")   # csr_matrix, shape=(3, vocab_size)

for i in range(len(docs)):
    # 第 i 行的非零权重值
    sparse_data = sparse_obj.data[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()
    # 这些权重分别落在哪些维度（token id）
    sparse_indices = sparse_obj.indices[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()
    # 拼成 {维度id: 权重}
    sparse_dict = {dim: weight for dim, weight in zip(sparse_indices, sparse_data)}
    sparse.append(sparse_dict)

print("稠密向量条数:", len(dense_), "每条维度:", len(dense_[0]))
print("稀疏向量条数:", len(sparse))
print("第 0 个文档的稀疏向量(前几项):", dict(list(sparse[0].items())[:5]))