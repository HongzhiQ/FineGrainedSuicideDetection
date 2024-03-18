# -*- coding: utf-8 -*-

import pandas as pd

def process_file(input_filename, output_filename):
    # Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=['comment', 'myLabel'])

    with open(input_filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Extract the text content of each line (removing serial numbers and dots)
            text = line.split('. ', 1)[1].strip() if '. ' in line else line.strip()

            # Add text and label 1 to DataFrame
            df = df._append({'comment': text, 'myLabel': 1}, ignore_index=True)
    # 保存DataFrame到TSV文件
    df.to_csv(output_filename, sep='\t', index=False, mode='a', header=False)

# 调用函数处理文件                                            1  2  3 4
process_file('source data', 'new generate data')
