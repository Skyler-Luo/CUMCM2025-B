#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
xlsx_to_csv.py - 将Excel XLSX文件转换为CSV格式
"""

import pandas as pd
import os
import argparse
import sys
from tqdm import tqdm


def xlsx_to_csv(input_file, output_file=None, sheet_name=0, encoding='utf-8', sep=',', index=False):
    """
    将XLSX文件转换为CSV格式
    
    参数:
        input_file (str): 输入的XLSX文件路径
        output_file (str): 输出的CSV文件路径，默认为与输入文件同名但后缀为.csv
        sheet_name (str|int): 要转换的表格名称或索引，默认为第一个表格
        encoding (str): 输出CSV文件的编码，默认为utf-8
        sep (str): CSV文件的分隔符，默认为逗号
        index (bool): 是否保留行索引，默认为False
    
    返回:
        bool: 转换成功返回True，失败返回False
    """
    try:
        # 如果未指定输出文件，则使用与输入文件相同的名称但更改扩展名
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.csv"

        # 读取Excel文件
        print(f"正在读取文件: {input_file}")
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 文件 '{input_file}' 不存在")
            return False
            
        # 获取所有表格名称
        xl = pd.ExcelFile(input_file)
        sheets = xl.sheet_names
        
        if isinstance(sheet_name, int) and (sheet_name >= len(sheets) or sheet_name < 0):
            print(f"错误: 表格索引 {sheet_name} 超出范围，文件包含 {len(sheets)} 个表格")
            print(f"可用的表格: {sheets}")
            return False
        elif isinstance(sheet_name, str) and sheet_name not in sheets:
            print(f"错误: 表格名称 '{sheet_name}' 不存在")
            print(f"可用的表格: {sheets}")
            return False
        
        # 读取指定的表格
        print(f"读取表格: {sheet_name}")
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        
        # 转换为CSV
        print(f"将数据转换为CSV并保存到: {output_file}")
        df.to_csv(output_file, encoding=encoding, sep=sep, index=index)
        
        print(f"转换完成! 共处理 {len(df)} 行数据")
        return True
        
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将Excel XLSX文件转换为CSV格式')
    parser.add_argument('input_file', help='输入的XLSX文件路径')
    parser.add_argument('-o', '--output', help='输出的CSV文件路径 (默认: 与输入文件同名但后缀为.csv)')
    parser.add_argument('-s', '--sheet', default=0, help='要转换的表格名称或索引 (默认: 0，第一个表格)')
    parser.add_argument('-e', '--encoding', default='utf-8', help='输出CSV文件的编码 (默认: utf-8)')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV文件的分隔符 (默认: 逗号)')
    parser.add_argument('--with-index', action='store_true', help='保留行索引 (默认: 不保留)')
    parser.add_argument('-l', '--list', action='store_true', help='列出XLSX文件中的所有表格并退出')
    
    args = parser.parse_args()
    
    # 如果只需列出表格
    if args.list:
        try:
            if not os.path.exists(args.input_file):
                print(f"错误: 文件 '{args.input_file}' 不存在")
                return 1
                
            xl = pd.ExcelFile(args.input_file)
            sheets = xl.sheet_names
            print(f"文件 '{args.input_file}' 包含以下表格:")
            for i, sheet in enumerate(sheets):
                print(f"  {i}: {sheet}")
            return 0
        except Exception as e:
            print(f"列出表格时出错: {str(e)}")
            return 1
    
    # 尝试将sheet参数转换为整数
    sheet_name = args.sheet
    try:
        sheet_name = int(args.sheet)
    except ValueError:
        # 如果转换失败，保持原始字符串
        pass
    
    # 执行转换
    success = xlsx_to_csv(
        args.input_file, 
        args.output, 
        sheet_name=sheet_name,
        encoding=args.encoding, 
        sep=args.delimiter,
        index=args.with_index
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
