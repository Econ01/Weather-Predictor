"""
Simple neural network architecture diagram using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11, 'Weather Temperature Prediction Model Architecture',
        ha='center', fontsize=16, fontweight='bold')

# Input Layer
input_box = patches.FancyBboxPatch((0.5, 8.5), 1.5, 2,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 9.8, 'INPUT', ha='center', fontweight='bold')
ax.text(1.25, 9.4, '30 days', ha='center', fontsize=9)
ax.text(1.25, 9.1, '12 features', ha='center', fontsize=9)

# Encoder
encoder_box = patches.FancyBboxPatch((2.5, 8.5), 1.8, 2,
                                     boxstyle="round,pad=0.1",
                                     edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(encoder_box)
ax.text(3.4, 10, 'ENCODER', ha='center', fontweight='bold')
ax.text(3.4, 9.5, '2-Layer GRU', ha='center', fontsize=9)
ax.text(3.4, 9.2, '256 hidden', ha='center', fontsize=9)
ax.text(3.4, 8.9, 'Dropout: 0.2', ha='center', fontsize=9)

# Arrow: Input -> Encoder
ax.annotate('', xy=(2.5, 9.5), xytext=(2.0, 9.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Encoder outputs
encoder_out1 = patches.FancyBboxPatch((4.8, 9.8), 1.2, 0.7,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='orange', facecolor='lightyellow', linewidth=1.5)
ax.add_patch(encoder_out1)
ax.text(5.4, 10.15, 'Hidden States', ha='center', fontsize=8, fontweight='bold')

encoder_out2 = patches.FancyBboxPatch((4.8, 8.8), 1.2, 0.7,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='orange', facecolor='lightyellow', linewidth=1.5)
ax.add_patch(encoder_out2)
ax.text(5.4, 9.15, 'All Outputs', ha='center', fontsize=8, fontweight='bold')

# Arrows: Encoder -> Outputs
ax.annotate('', xy=(4.8, 10.15), xytext=(4.3, 10),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
ax.annotate('', xy=(4.8, 9.15), xytext=(4.3, 9.2),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))

# Attention Mechanism
attention_box = patches.FancyBboxPatch((6.5, 8.5), 1.5, 2,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(attention_box)
ax.text(7.25, 10, 'ATTENTION', ha='center', fontweight='bold')
ax.text(7.25, 9.5, 'Bahdanau', ha='center', fontsize=9)
ax.text(7.25, 9.2, 'Mechanism', ha='center', fontsize=9)
ax.text(7.25, 8.9, 'Context Vector', ha='center', fontsize=8, style='italic')

# Arrows: Encoder outputs -> Attention
ax.annotate('', xy=(6.5, 10.15), xytext=(6.0, 10.15),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
ax.annotate('', xy=(6.5, 9.15), xytext=(6.0, 9.15),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))

# Decoder
decoder_box = patches.FancyBboxPatch((3.5, 5.5), 2, 2,
                                     boxstyle="round,pad=0.1",
                                     edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(decoder_box)
ax.text(4.5, 7, 'DECODER', ha='center', fontweight='bold')
ax.text(4.5, 6.5, 'Autoregressive', ha='center', fontsize=9)
ax.text(4.5, 6.2, '2-Layer GRU', ha='center', fontsize=9)
ax.text(4.5, 5.9, '256 hidden', ha='center', fontsize=9)

# Arrow: Attention -> Decoder
ax.annotate('', xy=(5.2, 7.5), xytext=(6.5, 8.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text(6, 8, 'context', ha='center', fontsize=8, style='italic', color='purple')

# Arrow: Hidden State -> Decoder
ax.annotate('', xy=(4.2, 7.5), xytext=(5.0, 9.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax.text(4.3, 8.5, 'hidden', ha='center', fontsize=8, style='italic', color='orange')

# Autoregressive loop
loop_arrow = patches.FancyArrowPatch((3.5, 6.5), (3.3, 6.5),
                                     connectionstyle="arc3,rad=0.3",
                                     arrowstyle='->', lw=2, color='darkred')
ax.add_patch(loop_arrow)
ax.text(2.8, 6.5, 'feedback', ha='center', fontsize=8, style='italic', color='darkred')

# Output Layer
output_box = patches.FancyBboxPatch((4, 2.5), 1, 2,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax.add_patch(output_box)
ax.text(4.5, 4, 'OUTPUT', ha='center', fontweight='bold')
ax.text(4.5, 3.5, '7 days', ha='center', fontsize=9)
ax.text(4.5, 3.2, 'Temperature', ha='center', fontsize=9)
ax.text(4.5, 2.9, 'Forecast', ha='center', fontsize=9)

# Arrow: Decoder -> Output
ax.annotate('', xy=(4.5, 4.5), xytext=(4.5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Add legend/info box
info_text = """Model Details:
• Input: 30 days × 12 features
• Hidden: 256 units
• Layers: 2 (Encoder + Decoder)
• Dropout: 0.2
• Parameters: 1.5M
• Output: 7-day forecast

Features Used:
TG, TN, TX, RR, SS, HU,
FG, FX, CC, SD,
DAY_SIN, DAY_COS"""

ax.text(8.5, 5, info_text, ha='left', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add data flow annotations
ax.text(5, 11.5, 'Data Flow: Input → Encoder → Attention → Decoder (with autoregressive feedback) → Output',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('model_architecture_diagram.png', dpi=300, bbox_inches='tight')
print("Model architecture diagram saved to model_architecture_diagram.png")
plt.show()
