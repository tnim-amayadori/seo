import datetime
import logging
import os

_folder_path: str = ""


# 中間生成ファイルを格納するフォルダを日付（yyyy-mmdd_hhmm-s）を名称としてdataフォルダ内に生成
def _create_data_folder(path_data):
    folder_name = datetime.datetime.now().strftime("%Y-%m%d_%H%M-%S")
    folder_path = os.path.join(path_data, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# ログ設定
def _setup_logging(folder_path):
    log_file = os.path.join(folder_path, f"{os.path.basename(folder_path)}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s[%(levelname)s]%(filename)s:%(lineno)d %(funcName)s():%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def initialize_logging(path_data):
    global _folder_path
    _folder_path = _create_data_folder(path_data)
    _setup_logging(_folder_path)


def get_daily_path(file_name: str):
    file_path = os.path.join(_folder_path, file_name)
    return file_path
