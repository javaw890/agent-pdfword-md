#全局初始化日志  全局你都能用
import logging
import colorlog

logger = logging.getLogger()
#级别 ≥ 阈值的才放行
logger.setLevel(logging.INFO)

# filename文件来源
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(

    '%(log_color)s%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))
#不用这个了 因为之前setuplogging会重新去新增一个addhandler 现在我是全局的不用了
# logger.handlers.clear()
logger.addHandler(handler)