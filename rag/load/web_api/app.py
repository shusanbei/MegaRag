import json
import os
from base64 import b64encode
from glob import glob
from io import StringIO
import tempfile
from typing import Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from magic_pdf.data.read_api import read_local_images, read_local_office
import magic_pdf.model as model_config
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.libs.config_reader import get_bucket_name, get_s3_config
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from fastapi import Form

# 设置使用内部模型
model_config.__use_inside_model__ = True

# 初始化FastAPI应用
app = FastAPI()

# 定义支持的文件扩展名
pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]

class MemoryDataWriter(DataWriter):
    """内存数据写入器，用于在内存中存储处理结果而不写入文件"""
    def __init__(self):
        self.buffer = StringIO()

    def write(self, path: str, data: bytes) -> None:
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            self.buffer.write(data.decode("utf-8"))

    def write_string(self, path: str, data: str) -> None:
        self.buffer.write(data)

    def get_value(self) -> str:
        return self.buffer.getvalue()

    def close(self):
        self.buffer.close()


def init_writers(
    file_path: str = None,
    file: UploadFile = None,
    output_path: str = None,
    output_image_path: str = None,
) -> Tuple[
    Union[S3DataWriter, FileBasedDataWriter],
    Union[S3DataWriter, FileBasedDataWriter],
    bytes,
]:
    """
    根据文件路径类型初始化写入器

    Args:
        file_path: 文件路径(本地路径或S3路径)
        file: 上传的文件对象
        output_path: 输出目录路径
        output_image_path: 图像输出目录路径

    Returns:
        Tuple[writer, image_writer, file_bytes, file_extension]: 返回初始化的写入器元组、文件内容和文件扩展名
    """
    file_extension:str = None
    if file_path:
        is_s3_path = file_path.startswith("s3://")
        if is_s3_path:
            # 处理S3路径
            bucket = get_bucket_name(file_path)
            ak, sk, endpoint = get_s3_config(bucket)

            writer = S3DataWriter(
                output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            image_writer = S3DataWriter(
                output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            # 临时创建reader读取文件内容
            temp_reader = S3DataReader(
                "", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            file_bytes = temp_reader.read(file_path)
            file_extension = os.path.splitext(file_path)[1]
        else:
            # 处理本地路径
            writer = FileBasedDataWriter(output_path)
            image_writer = FileBasedDataWriter(output_image_path)
            os.makedirs(output_image_path, exist_ok=True)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_extension = os.path.splitext(file_path)[1]
    else:
        # 处理上传的文件
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1]

        writer = FileBasedDataWriter(output_path)
        image_writer = FileBasedDataWriter(output_image_path)
        os.makedirs(output_image_path, exist_ok=True)

    return writer, image_writer, file_bytes, file_extension


def process_file(
    file_bytes: bytes,
    file_extension: str,
    parse_method: str,
    image_writer: Union[S3DataWriter, FileBasedDataWriter],
) -> Tuple[InferenceResult, PipeResult]:
    """
    处理文件内容

    Args:
        file_bytes: 文件的二进制内容
        file_extension: 文件扩展名
        parse_method: 解析方法('ocr', 'txt', 'auto')
        image_writer: 图像写入器

    Returns:
        Tuple[InferenceResult, PipeResult]: 返回推理结果和管道处理结果
    """

    ds: Union[PymuDocDataset, ImageDataset] = None
    if file_extension in pdf_extensions:
        # 处理PDF文件
        ds = PymuDocDataset(file_bytes)
    elif file_extension in office_extensions:
        # 处理Office文件
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_office(temp_dir)[0]
    elif file_extension in image_extensions:
        # 处理图像文件
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_images(temp_dir)[0]
    infer_result: InferenceResult = None
    pipe_result: PipeResult = None

    # 根据解析方法处理文件
    if parse_method == "ocr":
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    elif parse_method == "txt":
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    else:  # auto
        # 自动选择解析方法
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

    return infer_result, pipe_result


def encode_image(image_path: str) -> str:
    """使用base64编码图像"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


@app.post(
    "/file_parse",
    tags=["projects"],
    summary="解析文件(支持本地文件和S3)",
)
# async def file_parse(
def file_parse(
    file: UploadFile = None,
    file_path: str = Form(None),
    parse_method: str = Form("auto"),
    is_json_md_dump: bool = Form(False),
    output_dir: str = Form("uploads"),
    return_layout: bool = Form(False),
    return_info: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
):
    """
    执行将PDF转换为JSON和MD的过程，将MD和JSON文件输出到指定目录。

    Args:
        file: 要解析的文件。不能与`file_path`同时指定
        file_path: 要解析的文件路径。不能与`file`同时指定
        parse_method: 解析方法，可以是auto、ocr或txt。默认为auto。如果结果不理想，尝试ocr
        is_json_md_dump: 是否将解析的数据写入.json和.md文件。默认为False。不同阶段的数据将写入不同的.json文件(共3个)，md内容将保存到.md文件
        output_dir: 结果的输出目录。将创建一个以文件名命名的文件夹来存储所有结果
        return_layout: 是否返回解析的文件布局。默认为False
        return_info: 是否返回解析的文件信息。默认为False
        return_content_list: 是否返回解析的文件内容列表。默认为False
    """
    try:
        # 验证输入参数
        if (file is None and file_path is None) or (
            file is not None and file_path is not None
        ):
            return JSONResponse(
                content={"error": "必须提供file或file_path之一"},
                status_code=400,
            )

        # 获取文件名(不包含扩展名)
        file_name = os.path.basename(file_path if file_path else file.filename).split(
            "."
        )[0]
        # output_path = f"{output_dir}/{file_name}"
        output_path = f"uploads/{file_name}"
        output_image_path = f"{output_path}/images"

        # 初始化读取器/写入器并获取文件内容
        writer, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=file,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        # 处理文件
        infer_result, pipe_result = process_file(file_bytes, file_extension, parse_method, image_writer)

        # 使用内存写入器获取结果
        content_list_writer = MemoryDataWriter()
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()

        # 使用PipeResult的dump方法获取数据
        pipe_result.dump_content_list(content_list_writer, "", "images")
        pipe_result.dump_md(md_content_writer, "", "images")
        pipe_result.dump_middle_json(middle_json_writer, "")

        # 获取内容
        content_list = json.loads(content_list_writer.get_value())
        md_content = md_content_writer.get_value()
        middle_json = json.loads(middle_json_writer.get_value())
        model_json = infer_result.get_infer_res()

        # 如果需要保存结果
        if is_json_md_dump:
            # 注释掉的代码是保存详细结果的部分，默认不执行
            # writer.write_string(
            #     f"{file_name}_content_list.json", content_list_writer.get_value()
            # )
            writer.write_string(f"{file_name}.md", md_content)
            # writer.write_string(
            #     f"{file_name}_middle.json", middle_json_writer.get_value()
            # )
            # writer.write_string(
            #     f"{file_name}_model.json",
            #     json.dumps(model_json, indent=4, ensure_ascii=False),
            # )
            # # 保存可视化结果
            # pipe_result.draw_layout(os.path.join(output_path, f"{file_name}_layout.pdf"))
            # pipe_result.draw_span(os.path.join(output_path, f"{file_name}_spans.pdf"))
            # pipe_result.draw_line_sort(
            #     os.path.join(output_path, f"{file_name}_line_sort.pdf")
            # )
            # infer_result.draw_model(os.path.join(output_path, f"{file_name}_model.pdf"))

        # 构建返回数据
        data = {}
        if return_layout:
            data["layout"] = model_json
        if return_info:
            data["info"] = middle_json
        if return_content_list:
            data["content_list"] = content_list
        if return_images:
            image_paths = glob(f"{output_image_path}/*.jpg")
            data["images"] = {
                os.path.basename(
                    image_path
                ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                for image_path in image_paths
            }
        # data["md_content"] = md_content  # md_content始终返回
        data["file"] = f"{output_path}/{file_name}.md"  # 返回md文件路径

        # 清理内存写入器
        content_list_writer.close()
        md_content_writer.close()
        middle_json_writer.close()

        # return JSONResponse(data, status_code=200)
        return f"{output_path}/{file_name}.md"

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
