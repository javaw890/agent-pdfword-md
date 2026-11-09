'''
初始：import_file_path file文件的目录  file_dir文件的输出目录
entry:pdf_path文件的路径相当于改写了path(import_file_path)对象  is_md_read_enabled 标记是否为pdf  file_title 文件头
'''
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict

import requests



from conf.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exception import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        '''
        用来封装配置
        self.logger.info(self.config.vl_model)
        self.logger.info(mineru_config.base_url)
        self.logger.info(mineru_config.api_token)
        '''

        #1参数校验并返回path结果(说白了就是将上面那个节点的NodeEntry)
        print("111")
        pdf_path_obj,output_dir_obj = self._step_1_validate_paths(state)

        #2.将pdf上传到远程的miineru 并轮询结果是否解析完成 最后得到解压文件路径
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        print(zip_url)  #解压路径zip_url output_dir_obj目标输出路径 pdf_path_obj文件路径
        # 3.下载zip包      # 3.解压zip包
        md_path = self._step_3_generate_summaries(zip_url, output_dir_obj, pdf_path_obj.stem)


        # 4.将md文件读取出来
        with open(md_path,'r',encoding="utf-8")as f:
            md_content = f.read()

        #4.返回出来 解压后的内容
        state['md_content']=md_content
        #解压后的绝对路径
        state['md_path']=str(md_path)
        return state


    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """
        #1.参数的非空校验 有一个pdf路径 和输出路径
        pdf_path = state.get("pdf_path")
        #是否非空
        if not pdf_path:
            raise StateFieldError(field_name='pdf_path', expected_type=str,message="PDF路径不能为空")
        file_dir=state.get("file_dir")
        if not file_dir:
            raise  StateFieldError(field_name='file_dir', expected_type=str,message="输出路径不能为空")

        pdf_path_obj=Path(pdf_path)
        file_dir_obj=Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF文件{pdf_path_obj.name}不存在")

        #不存在递归创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录不存在，开始根据绝对路径自动创建：{file_dir_obj.absolute()}")
            #A B没创建 那就把这两个父子的层级都创建
            file_dir_obj.mkdir(parents=True, exist_ok=True)
        return pdf_path_obj, file_dir_obj
    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        #2.调用敏儿U远程api接口 实现文件上传远程连接服务器上 去找官方文档批量上传解析 先py ->申请文件上传连接-> 返回文件上传链接是url地址 是服务器连接
        #然后再上传进行解析
        token =self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        #pdf_path_obj.name和pdf_path_obj.stem就是.name就是文件名+后缀也包括了
        data = {
            "files": [

                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        file_path = [pdf_path_obj]
        #不用trycatch了因为外部就有
        response = requests.post(url, headers=header, json=data)
        #请求判断
        if response.status_code != 200:
            raise PdfConversionError('获取上传链接失败：response not success. status:{} ,result:{}'.format(response.status_code, response))
        #变成字典
        result = response.json()#获取响应结果
        #业务判断
        if result["code"] != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")
        #用来解析的结果
        batch_id = result["data"]["batch_id"]
        #这个是连接
        urls = result["data"]["file_urls"]
        self.logger.info(f"文件开始上传！")
        for i in range(0, len(urls)):
            with open(file_path[i], 'rb') as f:
                res_upload = requests.put(urls[i], data=f,headers={
                "Expect": "",
                "Content-Type": None,
            })
                if res_upload.status_code != 200:
                    raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
                self.logger.info(f"文件上传成功！")
        #获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"
        start_time=time.time()#记录当前系统时间
        timeout_seconds = 600  # 最大超时时间 s
        poll_interval = 3  # 轮询间隔时间 s
        self.log_step(step_name="轮询开始",
                             message=f"轮询间隔:{poll_interval}s,超时时间:{timeout_seconds}s,batch_id:{batch_id}")
        while True:
            #已消耗的时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，bactch_id：{batch_id}")
                self.logger.warning(f"异常信息，{str(e)}")
                #隔一下再试
                time.sleep(poll_interval)
                continue
            #数据层判断
            if res_poll.status_code!=200:
                raise PdfConversionError(f"【任务轮询】失败 状态码是:{res_poll.status_code}")
            poll_data = res_poll.json() #json->字典
            #轮询结果 业务层判断
            if poll_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]
            # 获取结果
            result_item = extract_results[0]
            data_state = result_item["state"]
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成,总耗时:{int(elapsed_time)}s,batch_id:{batch_id}"
                )
                full_zip_url = result_item["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功:{full_zip_url},总耗时:{int(elapsed_time)}s,batch_id:{batch_id}"
                )
                return full_zip_url

            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析处理中,总耗时:{elapsed_time},batch_id:{batch_id},状态:{data_state}"
                )
                time.sleep(poll_interval)


    def _step_3_generate_summaries(self, zip_url:str,output_dir_obj: Path, pdf_stem:str) -> str:
        """
        步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文

        :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要

        先拿zip 然后解压到对应地址+改名 拿里面的full.md 进行改名然后输出full.md的这个绝对路径
        """
        #就是发个 HTTP GET 请求，去 zip_url 那个地址把 zip 文件下载下来。如果返回的状态码不是 200（200 代表成功），就抛错，说下载失败。
        response = requests.get(zip_url)
        if response.status_code!=200:
            raise FileProcessingError(
                "下载失败"
            )
        #保存 zip 到本地拼一个保存路径，然后把下载下来的二进制数据写到这个 zip 文件里。
        zip_save_path=output_dir_obj/f"{pdf_stem}_result.zip"
        #把这个目标目录路径打开并且写入的意思 定义包的保存路径 定义一个路径然后让下载下来的zip放入里面 用对应的格式 并重新定义名字
        #创建一个新文件，把你给的二进制数据原封不动地写进去，一个字节都不改。
        with open(zip_save_path,'wb') as f:
            f.write(response.content)

        self.log_step("zip下载","下载成功")
        #解压 路径 拼一个解压目录的路径
        extra_target_dir=output_dir_obj/pdf_stem
        #删除已有目录 判断这个路径是否已经存在
        if extra_target_dir.exists():
            #先删除
            shutil.rmtree(extra_target_dir)
        #递归创建目录
        extra_target_dir.mkdir(parents=True,exist_ok=True)

        #解压zip 用只读模式打开刚才保存的那个 zip文件 把 zip 里面所有的内容，全部解压到 extra_target_dir 这个文件夹下面
        with zipfile.ZipFile(zip_save_path,'r') as zip_file:
            zip_file.extractall(extra_target_dir)
        self.log_step("zip解压","解压完成")

        target_md_file=extra_target_dir/"full.md"
        #把这个路径下的文件进行重命名
        new_md_path=target_md_file.with_name(f'{pdf_stem}.md')
        target_md_file.rename(new_md_path)
        return str(new_md_path.absolute())





if __name__=="__main__":
    init_state = {
        "pdf_path": r"E:\zhanggui\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\output"
    }
    setup_logging()

    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)
    json_dumps = json.dumps(result, ensure_ascii=False, indent=4)
    #因为只是调用了该节点所以没有之前的那些节点的state
    logging.getLogger().info(json_dumps)