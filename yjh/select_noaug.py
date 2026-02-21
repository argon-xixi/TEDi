import os
import shutil
import re

def filter_and_move_files(src_dir, dest_dir):
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 正则表达式：匹配文件名形如 "seq_a_framebbb.h5"（排除 _x 后缀）
    pattern = re.compile(r"^seq_\d{1,2}_frame\d{3}.h5$")

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        # 只筛选出形如 "seq_a_framebbb.h5" 的文件，排除 "_x" 后缀的文件
        if filename.endswith('.h5') and pattern.match(filename) and '_x' not in filename:
            # 构建源文件和目标文件的路径
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            
            # 将文件复制到新的目录
            shutil.copy(src_file, dest_file)
            print(f"Moved: {filename}")

# 输入源目录和目标目录路径
src_directory =  "/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_bina/train/" 
dest_directory = "/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_bina_noaug/train/"  # 请根据需要修改新的目标目录路径

filter_and_move_files(src_directory, dest_directory)
