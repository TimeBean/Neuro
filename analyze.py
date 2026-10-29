import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "results"
file_pattern = re.compile(r"e(\d+)l([\d\-]+)r([\d\.]+)\.txt")

data = []

for filename in os.listdir(results_dir):
    match = file_pattern.match(filename)
    if match:
        epochs = int(match.group(1))
        layers = match.group(2)
        lr = float(match.group(3))
        
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    
                    if ':' in last_line:
                        left_part, right_part = last_line.split(':')
                        
                        inputs = left_part.strip().split()
                        if len(inputs) == 2:
                            in1 = int(inputs[0])
                            in2 = int(inputs[1])
                            
                            target = in1 & in2 
                            prediction = float(right_part.strip())
                            error = abs(target - prediction)
                            
                            data.append({
                                "Epochs": epochs,
                                "Layers": layers,
                                "LearningRate": lr,
                                "Error": error,
                                "Prediction": prediction
                            })
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

df = pd.DataFrame(data)

if df.empty:
    print("No data loaded. Check the 'results' directory path.")
    exit()

logical_order = ["2-32-1", "2-64-1", "2-128-1", "2-64-32-1"]
ordered_layers = [l for l in logical_order if l in df['Layers'].unique()]
df['Layers'] = pd.Categorical(df['Layers'], categories=ordered_layers, ordered=True)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 10))

# Plot 1: Error vs Epochs
plt.subplot(2, 2, 1)
df_lr_fixed = df[df['LearningRate'] == 0.001]
if not df_lr_fixed.empty:
    sns.lineplot(data=df_lr_fixed, x='Epochs', y='Error', hue='Layers', marker='o')
    plt.title('Error Decrease Over Time (LR = 0.001)')
    plt.ylabel('Absolute Error')

# Plot 2: Error vs Learning Rate
plt.subplot(2, 2, 2)
df_epochs_fixed = df[df['Epochs'] == 50000]
if not df_epochs_fixed.empty:
    ax = sns.lineplot(data=df_epochs_fixed, x='LearningRate', y='Error', hue='Layers', marker='o')
    ax.set_xscale('log') 
    plt.title('Impact of Learning Rate on Error (at 50,000 Epochs)')
    plt.ylabel('Absolute Error')

# Plot 3: Heatmap
plt.subplot(2, 2, 3)
max_epochs = df['Epochs'].max()
df_max_epochs = df[df['Epochs'] == max_epochs]
if not df_max_epochs.empty:
    # Добавлен observed=False для исправления FutureWarning
    pivot = df_max_epochs.pivot_table(index='Layers', columns='LearningRate', values='Error', observed=False)
    sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', fmt=".4f", vmin=0, vmax=1)
    plt.title(f'Error Heatmap (closer to 0 is better)\nat {max_epochs} Epochs')

# Plot 4: Predictions vs Target
plt.subplot(2, 2, 4)
sns.stripplot(data=df, x='Layers', y='Prediction', hue='LearningRate', jitter=True, alpha=0.7)
plt.axhline(1.0, color='green', linestyle='--', label='Ideal Target (1.0)')
plt.title('Actual Network Predictions (Input 1 AND 1 = 1)')
plt.ylabel('Network Prediction')
plt.legend(title='Learning Rate')

plt.tight_layout()
plt.show()
