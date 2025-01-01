"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import os
from os.path import join
import pandas as pd

base_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/gelanalyzer'

for i in range(35):
    analysis = join(base_folder, str(i))
    if os.path.isdir(analysis):
        full_data = []
        for file in ['morphological.txt', 'valley.txt']:
            with open(join(analysis, file), 'r') as f:
                g_data = f.readlines()
            i = 0
            for line in g_data:
                if line.startswith('Lane'):
                    continue
                if line.startswith('\n'):
                    continue
                info = line.split('\t')
                if file == 'morphological.txt':
                    data_list = [int(info[0]), int(info[1]), int(info[4])]
                    full_data.append(data_list)
                else:
                    full_data[i].append(int(info[4]))
                i += 1

        df = pd.DataFrame(full_data, columns=['Lane ID', 'Band ID', 'Morphological Volume', 'Valley-to-Valley Volume'])
        df.to_csv(join(analysis, 'extra_background_data.csv'), index=False)

