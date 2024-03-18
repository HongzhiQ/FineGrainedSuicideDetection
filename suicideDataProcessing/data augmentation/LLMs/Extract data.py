# -*- coding: utf-8 -*-
import pandas as pd

def process_data(input_file, output_file, num_rows):
    # Load data from file
    data = pd.read_csv(input_file, sep='\t',encoding='utf-8')

    # Initialize an empty DataFrame for results
    result = pd.DataFrame()

    # The size of each group of data
    group_size = 3

    # Calculate how many groups there are in total
    num_groups = (len(data) + group_size - 1) // group_size

    # Open the file and prepare to append data
    with open(output_file, 'a', newline='') as file:
        # Extract the first data of each group, then the second, until the specified number of rows is reached
        for i in range(group_size):
            for j in range(num_groups):
                row_index = j * group_size + i

                # Stop if the row index exceeds the data range or the required number of rows has been reached
                if row_index >= len(data) or len(result) >= num_rows:
                    break

                #Add rows to results
                result = result._append(data.iloc[row_index])
                # Append the current line to the file
                data.iloc[[row_index]].to_csv(file, sep='\t', index=False, header=file.tell()==0)

    return result

input_file = ''
output_file = ''
num_rows = 291  ##Replace with the value of the new amount of data you need
processed_data = process_data(input_file, output_file, num_rows)
processed_data.head()
