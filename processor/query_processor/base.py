"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from abc import abstractmethod, ABC
from typing import TypeVar

from processor.query_processor.my_logging import logger
from processor.query_processor.state import QueryGraphState

T = TypeVar("T")  # 泛型状态类型
class NodeBase(ABC):

    name: str = "base_node"  # 跟节点名称，子类应覆盖

    def __call__(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点执行入口
        """
        try:
            # 1. 开始准备执行节点 建议别用print 一般用的是logging
            logger.info(f"--- {self.name} 开始啦 ---")

            # 2. 执行节点
            result = self.process(state)

            # 3. 执行节点成功
            logger.info(f"--- {self.name} 完成啦 ---")

            return result
        #异常跟踪栈
        except Exception as e:
            #唯一区别是 exception() 会自动把当前异常的完整堆栈（Traceback）附在消息后面一起打出来，error() 只打你写的那行文字，不带堆栈。
            logger.exception(f"{self.name} 执行失败: {e}")
            # logger.error(f"{self.name} 执行失败: {e}")
            #不用中断继续往下执行 降级处理 可以往下执行不中断
            # raise e

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        子类必须实现此方法
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        pass