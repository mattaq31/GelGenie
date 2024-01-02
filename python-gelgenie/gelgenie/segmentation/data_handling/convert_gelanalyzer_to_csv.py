import os
from os.path import join
import pandas as pd

base_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/gelanalyzer'

for i in range(34):
    analysis = join(base_folder, str(i))
    if os.path.isdir(analysis):
        full_data = []
        for file in ['uncorrected.txt', 'corrected.txt']:
            with open(join(analysis, file), 'r') as f:
                g_data = f.readlines()
            i = 0
            for line in g_data:
                if line.startswith('Lane'):
                    continue
                if line.startswith('\n'):
                    continue
                info = line.split('\t')
                if file == 'uncorrected.txt':
                    data_list = [int(info[0]), int(info[1]), int(info[4])]
                    full_data.append(data_list)
                else:
                    full_data[i].append(int(info[4]))
                i += 1

        df = pd.DataFrame(full_data, columns=['Lane ID', 'Band ID', 'Raw Volume', 'Background Corrected Volume'])
        df.to_csv(join(analysis, 'collated_data.csv'), index=False)

