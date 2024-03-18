# -*- coding: utf-8 -*-
import pandas as pd

def process_data(input_file, output_file, num_rows):
    # Load data from file
    data = pd.read_csv(input_file, sep='\t',encoding='GBK')

    # Initialize an empty DataFrame for results
    result = pd.DataFrame()

    # The size of each group of data
    group_size = 4

    # Calculate how many groups there are in total
    num_groups = (len(data) + group_size - 1) // group_size

    # Open the file and prepare to append data
    with open(output_file, 'a', newline='') as file:
        # Extract the first data of each group, then the second, until the specified number of rows is reached
        for i in range(group_size):
            for j in range(num_groups):
                row_index = j * group_size + i

                # Stop if row index exceeds data range, or required number of rows has been reached
                if row_index >= len(data) or len(result) >= num_rows:
                    break

                # Add rows to results
                result = result._append(data.iloc[row_index])
                # Append current line to file
                data.iloc[[row_index]].to_csv(file, sep='\t', index=False, header=file.tell()==0)

    return result

input_file = 'synonyms/New GenerateData/round1-train/tongyiciData-round1-10.tsv'
output_file = ''
num_rows = 132   #Replace with the value of the new amount of data you need
# This will process the data and save it in the specified otput_fie
processed_data = process_data(input_file, output_file, num_rows)
processed_data.head()
