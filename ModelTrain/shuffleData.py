import pandas as pd
import random

# Replace with your TSV file name
input_file = ''
output_file = ''

data = pd.read_csv(input_file, sep='\t')

# Shuffle only the data rows, excluding the header row
shuffled_data = data.iloc[1:].sample(frac=1).reset_index(drop=True)

# Re-add the header row to the top
shuffled_data = pd.concat([data.iloc[0:1], shuffled_data]).reset_index(drop=True)

# Save the shuffled data to a new file
shuffled_data.to_csv(output_file, sep='\t', index=False)
