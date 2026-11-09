from pymilvus import MilvusClient

from conf.milvus_config import milvus_config

_milvu_sclient=None
def get_milvu_sclient():
    global _milvu_sclient
    if _milvu_sclient is not None:
        return _milvu_sclient

    _milvu_sclient = MilvusClient(milvus_config.milvus_url)
    return _milvu_sclient