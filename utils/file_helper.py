import os
import re

def read_file_list(dir_path, pattern=r'.*', select_sub_path=True, is_full_path=True):
  '''读取文件列表'''
  # 文件列表
  files = []
  # 正则
  pattern = re.compile(pattern)
  # 查询子路径
  if select_sub_path==1:
    for dirpath, dirnames, filenames in os.walk(dir_path):
      for f in filenames:
        # 转小写
        file_name = f
        file_path = os.path.join(dirpath, f)
        file_path = file_path.replace('\\', '/')
        if not is_full_path:
          file_path = file_path.replace(dir_path.replace('\\', '/'), '')
          if file_path[0]=='/':
            file_path = file_path[1:]
        match = pattern.search(file_name.lower())
        if match:
          files.append(file_path)
  else:
    # 只查询当前目录文件
    for file_or_dir_name in os.listdir(dir_path):
      if os.path.isfile(file_or_dir_name):
        f = file_or_dir_name
        # 转小写
        file_name = f
        file_path=os.path.join(dir_path, f)
        if not is_full_path:
          file_path = file_path.replace(dir_path, '')
        match = pattern.search(file_name.lower())
        if match:
          files.append(file_path)
  return files


def read_dir_list(dir_path, pattern=r'.*', select_sub_path=True):
  '''读取文件夹列表'''
  # 文件列表
  files = []
  # 正则
  pattern = re.compile(pattern)
  # 查询子路径
  if select_sub_path==1:
    for dirpath, dirnames, filenames in os.walk(dir_path):
      for f in dirnames:
        # 转小写
        file_name = f.lower()
        file_path=os.path.join(dirpath, f)
        match = pattern.search(file_name)
        if match:
          files.append(file_path)
  else:
    # 只查询当前目录文件
    for file_or_dir_name in os.listdir(dir_path):
      if os.path.isdir(file_or_dir_name):
        f = file_or_dir_name
        # 转小写
        file_name = f.lower()
        file_path=os.path.join(dir_path, f)
        match = pattern.search(file_name)
        if match:
          files.append(file_path)
  return files
