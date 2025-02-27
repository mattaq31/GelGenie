import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023'
model = 'unet_dec_21'
df = pd.read_csv(os.path.join(model_folder, model, 'training_logs', 'training_stats.csv'))

# Create output directory for frames
frame_dir = "/Users/matt/Desktop/training_animation"
os.makedirs(frame_dir, exist_ok=True)

# Plot settings
c1 = '#FF9600'
c2 = '#3142AC'
line_width = 4
label_size = 60
tick_size = 35

sns.set(style="white")
plt.rcParams.update({'font.sans-serif': 'Helvetica'})
plt.rcParams['font.family'] = 'Helvetica'

# Generate frames
for i in range(1, len(df) + 1):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()
    
    # Set transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax2.patch.set_alpha(0)

    # Plot up to current frame
    sns.lineplot(x=df['Epoch'][:i], y=df['Training Loss'][:i], color=c1, linewidth=line_width, ax=ax, label='Training Loss')
    sns.lineplot(x=df['Epoch'][:i], y=df['Dice Score'][:i], color=c2, linestyle='dashed', linewidth=2, ax=ax2, label='Actual Trace')

    # Running average line
    running_average = df['Dice Score'].rolling(window=10).mean()
    sns.lineplot(x=df['Epoch'][:i], y=running_average[:i], color=c2, linewidth=line_width, ax=ax2, label='Running Average')

    # Set labels
    ax.set_xlabel('Epoch', fontsize=label_size, weight="bold")
    ax.set_ylabel('Training Loss', color=c1, fontsize=label_size, weight="bold")
    ax2.set_ylabel('Validation Score', color=c2, fontsize=label_size, weight="bold")
    ax.tick_params(axis='y', labelcolor=c1, labelsize=tick_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax2.tick_params(axis='y', labelcolor=c2, labelsize=tick_size)

    # Set limits so the view doesn't change
    ax.set_xlim(df['Epoch'].min(), df['Epoch'].max())
    ax.set_ylim(df['Training Loss'].min(), df['Training Loss'].max())
    ax2.set_ylim(df['Dice Score'].min(), df['Dice Score'].max())

    # Make legend transparent
    ax.legend([], [], frameon=False)
    legend = ax2.legend(fontsize=label_size, loc='center right', frameon=True)
    legend.get_frame().set_alpha(0)  # Fully transparent legend

    # Save frame with transparency
    frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
    plt.savefig(frame_path, dpi=300, transparent=True, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

