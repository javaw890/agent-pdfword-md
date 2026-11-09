#导入功能主流程封装边节点哪些
'''
workflow = KBImportWorkflow()
workflow.run(init_state, stream=False)去拿graph
graph里面去调用build_graph 里面去构建图 最后编译流即可
里面调用NodeEntry() ->call魔法方法
然后去调用基类baseBaseNode __call__为一个个节点（self是KBImportWorkflow 也有basenode）
'''
import logging

from langgraph.constants import END
from langgraph.graph import StateGraph


import json

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recogintion import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    #路由方法 类方法  这个是条件边函数
    @staticmethod
    def route_after_entry(state:ImportGraphState)->str:
        if state.get("is_pdf_read_enabled"):
            return 'node_pdf_to_md'
        elif state.get("is_md_read_enabled"):
            return 'node_md_img'
        else:
            return END
    def __init__(self):
        '''
        初始化工作流对象
        '''
        self.__compiled_graph=None

    #@property 是 Python 用来把一个类方法 "伪装成属性" 的装饰器
    @property
    def graph(self):
        '''
        延迟加载旨在第一次使用工作流对象时去编译图

        :return:
        '''
        if self.__compiled_graph is None:
            self.__compiled_graph=self.build_graph()
        return self.__compiled_graph


    def build_graph(self):
        #进行全局所以准确说法是：它是这张图所用状态的类型定义，而不是一个全局状态对象。 说它 "全局" 只在一个意义上成立：这个类型在模块里定义一次，整张图所有节点都共用同一份声明，不用每个节点重复写。
        graph=StateGraph(ImportGraphState)
        # 2. 注册节点到工作流 把类的（） 会调用call也就是继承的父类call作为节点 再调用子类的process 加给节点起别名
        graph.add_node("node_entry", NodeEntry()) #入口文件
        graph.add_node("node_pdf_to_md", NodePDFToMD())#用条件边去看md or pdf or其他格式 换位md
        graph.add_node("node_md_img", NodeMDImg()) #解析文档中的图片的含义并将其放入嵌入文档中
        graph.add_node("node_document_split", NodeDocumentSplit())#切分
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())#去进行去识别商品名称
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())#将文本转为向量
        graph.add_node("node_import_milvus", NodeImportMilvus())#将向量和标量导入milvus
        #定义一个条件边节点
        graph.set_entry_point("node_entry")

        #判断为pdf
        # 4. 注册条件边 条件会走他那边的（如果打印的话） 其他的边就不会显示了 根据哪个函数呢route_after_entry的返回值 去到哪个节点
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )

        # 5. 注册顺序边
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        # 6. 编译工作流 graph进行编译
        return graph.compile()


    def run(self,state:ImportGraphState,stream:bool=False):
        '''
        这个方法是对上面两个方法的启动了属于是
        :param state:
        :param stream:
        :return:
        '''
        #self=workflow实例化对象
        # #一开始时app
        # app = self.build_graph()
        # app.get_graph().print_ascii()


        # self.graph 去拿graph.compile()图对象 如果没有就会调用build_graph
        # 打印编译后的图
        # self.graph.get_graph().print_ascii()
        if stream:
            return self.graph.stream(state)
        else:
            return self.graph.invoke(state)



#测试一下
if __name__ == "__main__":
    #激活日志 五颜六色
    setup_logging()
    # 定义初始状态 这个是从start节点开始 如果是某个节点里面的return 也会组装起来
    init_state = {"import_file_path": r"D:\zhangguidoc\hak180使用说明书.pdf",
                  "file_dir": r"D:\output"
                  }

    # 方式1：实例化后使用（推荐方式，可复用）
    workflow = KBImportWorkflow()

    # for event in workflow.run(init_state, stream=True):
    #     print(f"state: {event}")

    # 方式2：非流式执行
    final_state = workflow.run(init_state, stream=False)

    #logging.getLogger() 不传参数 = 返回 root logger（根 logger）；
    logging.getLogger().info(final_state)
    #json.dumps(obj) = dump string，把一个 Python 对象（通常是 dict /list）序列化成一个 JSON 格式的字符串，返回值是个 str。 这是中文场景下几乎必须加的参数。 缩进4个字符
    logging.getLogger().info(json.dumps(final_state, ensure_ascii=False, indent=4,default=str))
    #这玩意儿直接把 ASCII 图打到控制台，返回值是 None
    # workflow.graph.get_graph().print_ascii() draw_ascii() 不打印，返回 ASCII 字符串；
    logging.getLogger().info(workflow.graph.get_graph().draw_ascii())
