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

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


out_folder = '/Users/matt/Desktop'
model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023'
model = 'unet_dec_21'

df = pd.read_csv(os.path.join(model_folder, model, 'training_logs', 'training_stats.csv'))

c1 = 'tab:red'
c2 = 'tab:blue'
c3 = 'tab:green'
c4 = 'tab:purple'

label_size = 40
tick_size = 26
legend_size = 35
line_width = 4

sns.set(style="white")
plt.rcParams.update({'font.sans-serif': 'Helvetica'})
plt.rcParams['font.family'] = 'Helvetica'

fig = plt.figure(figsize=(20, 10))
ax = sns.lineplot(x='Epoch', y='Training Loss', data=df, color=c1, linewidth=line_width, label='Training Loss')
ax2 = ax.twinx()

[x.set_linewidth(2.5) for x in ax.spines.values()]

sns.lineplot(x='Epoch', y='Dice Score', data=df, label='Actual Trace', linestyle='dashed',
             color=c2, linewidth=2, ax=ax2)

# adds more lines for component loss values
# sns.lineplot(ax=ax, x='Epoch', y='Dice Loss', data=df, color=c3, linewidth=line_width, label='Dice Loss')
# sns.lineplot(ax=ax, x='Epoch', y='Cross-Entropy Loss', data=df, color=c4, linewidth=line_width, label='Cross-Entropy Loss')

running_average = df['Dice Score'].rolling(window=10).mean()

sns.lineplot(x=df['Epoch'], y=running_average, color=c2, linewidth=line_width,
             label='Running Average', ax=ax2)

# aligns both tick labels together
# ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 7))
# ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

ax.set_ylabel('Two-Component Training Loss', color=c1, fontsize=label_size, weight="bold")
ax.set_xlabel('Epoch', fontsize=label_size, weight="bold")
ax.tick_params(axis='y', labelcolor=c1, labelsize=tick_size)
ax.tick_params(axis='x', labelsize=tick_size)

ax2.set_ylabel('Validation Dice Score', color=c2, fontsize=label_size, weight="bold")
ax2.tick_params(axis='y', labelcolor=c2, labelsize=tick_size)

ax.legend([], [], frameon=False)
ax2.legend(fontsize=legend_size, loc='center right')
# ax2.legend([], [], frameon=False)

plt.savefig(os.path.join(out_folder, 'example_training_plot.png'), bbox_inches='tight', dpi=300)
plt.show()
