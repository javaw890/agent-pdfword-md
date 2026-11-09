"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能


、typing 里和它并列、日常高频出现的类型工具
Optional 只是类型标注世界里的一个小构件，同层常用的还有这些：
1. 联合类型本体 Union
Union[A, B, C] 表示 "是 A、B、C 中的任意一种"。Optional 只是 Union[X, None] 的特化。3.10 起可以直接写 A | B | C。
2. Literal（限定取值枚举）
Literal["get", "post", "delete"]，表示只能是这几个字符串字面量之一，而不是任意 str。和 Optional 组合：Optional[Literal["get", "post"]] 表示要么传这两个值，要么 None。
3. 容器泛型
现代 Python（3.9+）直接用小写内建类型带参数：list[int]、dict[str, ImportConfig]、set[float]、tuple[int, int]。老写法是 List[int]、Dict[str, X]、Tuple[int, int]（从 typing 导入），本质一样。
4. 抽象容器（只要求行为，不要求具体实现）
Sequence[int]：能按索引取、可迭代，不要求一定是 list（tuple、range 也算）。
Iterable[int]：只要能 for 循环遍历即可，最宽松。
Mapping[str, int]：只读字典协议，函数体里不修改时优先用它，而不是 dict。
5. Callable（函数作为参数）
Callable[[int, str], bool]，方括号里前一个列表是参数类型，最后一个是返回值类型，表示 "接收 int 和 str、返回 bool 的函数"。
6. Any、NoReturn / Never
Any：完全不约束，等于放弃类型检查，能和任何类型互通。
NoReturn：这个函数永远不返回（抛异常、死循环），3.11 起又叫 Never。
7. Type[T]
Type[ImportConfig] 标注的是 "类本身"，不是实例。对比：ImportConfig 是实例，Type[ImportConfig] 是类对象本身（用来当构造器传）。
8. Final、ClassVar、Annotated
Final[int]：赋值后不许再改，声明常量。
ClassVar[int]：类属性，不属于实例。
Annotated[ImportConfig, "额外元信息"]：在类型上挂附加说明（Pydantic 的 Field、FastAPI 的依赖注入经常走这个通道）。
9. TypeVar、Generic、Protocol
这三个是写泛型库时才碰的：TypeVar 定义类型变量，Generic[T] 定义泛型类，Protocol 定义结构化接口（鸭子类型的静态版本）。普通业务代码很少直接写。
三、那为啥不直接写 None | ImportConfig
短答案：现在（3.10+）完全可以直接写，两者等价，新代码没有任何理由非用 Optional；你看到满屏 Optional 主要是历史遗留和兼容性。 展开说有三个原因。
"""
import colorlog

from processor.import_processor.config import ImportConfig, get_config
from processor.import_processor.exception import ImportProcessError
from utils.task_utils import add_done_task, add_running_task

"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

T = TypeVar("T")  # 泛型状态类型

#节点的抽象父类因为有ABC 这个就是说需要 前面print 中间print+调用函数 后面print
class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: ImportConfig|None = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        #已经将这个logging起名了 后续self.logger就是用了个名字的logging
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        参数是什么类型返回也是对应的类型
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。
        这个是类（）就会调用这个魔法方法给到子类 用来去环绕 全部给到子类  子类去继承所有
        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        try:
            # 1. 开始准备执行节点 logger.info去记录日志
            self.logger.info(f"--- {self.name} 开始 ---")
            # 开始：记录节点运行状态
            add_running_task(state["task_id"], self.name)
            # 2. 执行节点 实例对象（）去调用init
            #对，self 永远是实际创建出来的那个实例——也就是最外层子类的实例。整个继承链里从头到尾只有一个对象，父类从来不会额外“造一个父对象”
            result = self.process(state)
            # 结束：记录节点完成状态
            add_done_task(state["task_id"], self.name)
            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                #相当于先捕获Exception然后再把异常定住不要往外走
                cause=e
            )
    #@abstractmethod 用来强制子类必须实现某个方法——它是给基类作者立规矩的工具：声明“接口在这，子类不实现就不许实例化”
    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志
        就是把 " 拼一个 [步骤名] 前缀、可选再附一句话、然后交给 logger.info 打出去 " 这件重复的小事封装成一个方法。说白了就是套了个统一格式的外壳。
        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式 进行初始化logging
def setup_logging(level: int = logging.INFO):
    """
    配置日志格式

    Args:
        level: 日志级别
    设置level+一些颜色的设置
    StreamHandler 管 "打到控制台"，ColoredFormatter 管 "每行长啥样、什么颜色"，clear () 防止重复挂导致重复打印，addHandler 把这套渠道接上去。
    logger.handlers 是一个列表，addHandler 只会往列表末尾追加，不会帮你去重；不清空就会越堆越多，同一条日志被每个 handler 各打印一遍。
    假设你的配置代码被执行了 3 次（后面讲为什么会执行多次）：
第 1 次：addHandler(h1)，列表变成 [h1]，一条日志打 1 遍。
第 2 次如果不清空直接 addHandler(h2)，列表变成 [h1, h2]，同一条日志 h1 打一遍、h2 再打一遍，屏幕上出现 2 遍。
第 3 次再加 h3，列表 [h1, h2, h3]，同一条日志打 3 遍。
    """



    logger = logging.getLogger()
    logger.setLevel(level)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',  # INFO 显示为绿色
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))

    logger.handlers.clear()
    logger.addHandler(handler)
