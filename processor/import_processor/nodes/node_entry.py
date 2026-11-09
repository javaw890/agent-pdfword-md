import json
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exception import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    判断文件拓展名是pdf ormd
    如果是md则return {“is_md_read_enabled:True”}
    如果是pdf return {“is_pdf_read_enabled:True”}
    如果都不是 则后面会抛异常即可 因为俩个都是false
    """


    name = "node_entry"

    def process(self, state: ImportGraphState):
        '''
        processor.import_processor.exception.ImportProcessError: [node_entry] 'import_file_path' (原因: 'import_file_path')
        相当于拼拼拼 然后封装成ImportProcessError
        '''
        #state中获取获取用户传的文件路径
        # import_file_path_ = state['import_file_path']
        import_file_path_ = state.get('import_file_path')
        #判断路径是否为空
        if not import_file_path_:
            #有一个知识点就是state.get(a) 和state['a']这两者区别就在state['a']如果没有会直接报错Keyerror
            #当这个异常一直没有捕获就会抛到外面然后就是调用该异常的__str__
            #后面ImportProcessError这里面tryexcept去捕获StateFieldError然后里面又raise去类似于封装的意思 然后拼拼拼就完事了 把e=StateFieldError也就是cause 也调用str会继承基类
            raise StateFieldError(field_name="import_file_path",message="文件路径不能为空"
                                  ,expected_type=str,
                                  node_name=self.name)

            # 2.转为path对象 将file_path转为path对象
        #转为path对象 后面可以有名字 后缀那些相当于编程一个类
        import_file_path_obj = Path(import_file_path_)
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f'文件{import_file_path_obj.name}不存在')

        # 3检查后缀
        if import_file_path_obj.suffix==".pdf":
            state["is_pdf_read_enabled"]=True
            state['pdf_path']=import_file_path_obj
        elif import_file_path_obj.suffix=='.md':
            state["is_md_read_enabled"] = True
            state['md_path'] = import_file_path_obj
        else:
            raise ValidationError(message=f"不支持文件类型{import_file_path_obj.suffix}")



        #获取文件名作为标题
        state['file_title']=import_file_path_obj.stem
        #这里为什么返回的是state 而不是要啥给啥呢 因为此时不会出现污染问题后续不会被污染
        #把文件名然后文件路径 然后是否是md or pdf给一个属性确认后续用来做条件边
        return state





        #用来前面测试basenode的
        # return {
        #     "is_md_read_enabled":True,
        #     "is_pdf_read_enabled":False
        # }
        # return {
        #     "is_md_read_enabled": False,
        #     "is_pdf_read_enabled": True
        # }
        # return {
        #     "is_md_read_enabled": False,
        #     "is_pdf_read_enabled": False
        # }


#单测 测试这个节点
if __name__=="__main__":
    logging1 = setup_logging()
    print(logging1)
    init_state={
        "import_file_path":r"D:\output\hak180产品安全手册\hak180产品安全手册.pdf"
    }
    node_entry=NodeEntry()
    #调用call函数
    res = node_entry(init_state)
    print("111")
    print(res)
    logging.getLogger().info(json.dumps(res,default=str, ensure_ascii=False, indent=4))

