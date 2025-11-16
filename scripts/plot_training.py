"""
Training Plot Generation Script

This script generates comprehensive training plots from YOLO training results CSV file.
It creates loss curves, mAP curves, precision/recall curves, and learning rate plots.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import ensure_dir


def plot_training_curves(results_csv_path: str, output_dir: str = None):
    """
    Generate comprehensive training plots from results CSV file.
    
    Args:
        results_csv_path: Path to the results.csv file from YOLO training
        output_dir: Directory to save plots (default: same directory as CSV)
    """
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Results CSV file not found: {results_csv_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_csv_path)
    # Create directory (handle Windows encoding issues)
    try:
        ensure_dir(output_dir)
    except UnicodeEncodeError:
        # Fallback: create directory without printing
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading training results from: {results_csv_path}")
    df = pd.read_csv(results_csv_path)
    
    if df.empty:
        raise ValueError("Results CSV file is empty")
    
    epochs = df['epoch'].values
    
    # Set style for better-looking plots (try different style names for compatibility)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            try:
                plt.style.use('dark_background')
            except OSError:
                pass  # Use default style
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss Curves (Training and Validation)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, df['train/box_loss'], label='Train Box Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, df['val/box_loss'], label='Val Box Loss', linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Box Loss', fontsize=11)
    ax1.set_title('Box Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, df['train/cls_loss'], label='Train Class Loss', linewidth=2, alpha=0.8, color='orange')
    ax2.plot(epochs, df['val/cls_loss'], label='Val Class Loss', linewidth=2, alpha=0.8, 
             linestyle='--', color='red')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Class Loss', fontsize=11)
    ax2.set_title('Class Loss Curves', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, df['train/dfl_loss'], label='Train DFL Loss', linewidth=2, alpha=0.8, color='green')
    ax3.plot(epochs, df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, alpha=0.8, 
             linestyle='--', color='darkgreen')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('DFL Loss', fontsize=11)
    ax3.set_title('DFL Loss Curves', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 2. mAP Curves
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', linewidth=2.5, alpha=0.9, color='blue')
    ax4.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2.5, alpha=0.9, color='purple')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('mAP', fontsize=11)
    ax4.set_title('Mean Average Precision (mAP) Curves', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 3. Precision and Recall
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, df['metrics/precision(B)'], label='Precision', linewidth=2.5, alpha=0.9, color='coral')
    ax5.plot(epochs, df['metrics/recall(B)'], label='Recall', linewidth=2.5, alpha=0.9, color='teal')
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title('Precision & Recall Curves', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 4. Learning Rate
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, df['lr/pg0'], label='LR pg0', linewidth=2, alpha=0.8, color='darkblue')
    ax6.plot(epochs, df['lr/pg1'], label='LR pg1', linewidth=2, alpha=0.8, color='darkgreen')
    ax6.plot(epochs, df['lr/pg2'], label='LR pg2', linewidth=2, alpha=0.8, color='darkred')
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Learning Rate', fontsize=11)
    ax6.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    try:
        print(f"✓ Comprehensive training curves saved to: {output_path}")
    except UnicodeEncodeError:
        print(f"[OK] Comprehensive training curves saved to: {output_path}")
    plt.close()
    
    # Create individual plots for better visibility
    
    # 1. Combined Loss Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train/box_loss'], label='Train Box Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, df['train/cls_loss'], label='Train Class Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, df['train/dfl_loss'], label='Train DFL Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, df['val/box_loss'], label='Val Box Loss', linewidth=2, alpha=0.8, linestyle='--')
    ax.plot(epochs, df['val/cls_loss'], label='Val Class Loss', linewidth=2, alpha=0.8, linestyle='--')
    ax.plot(epochs, df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    try:
        print(f"✓ Loss curves saved to: {loss_path}")
    except UnicodeEncodeError:
        print(f"[OK] Loss curves saved to: {loss_path}")
    plt.close()
    
    # 2. mAP Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', linewidth=3, alpha=0.9, color='blue')
    ax.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=3, alpha=0.9, color='purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Mean Average Precision (mAP) Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    map_path = os.path.join(output_dir, 'map_curves.png')
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    try:
        print(f"✓ mAP curves saved to: {map_path}")
    except UnicodeEncodeError:
        print(f"[OK] mAP curves saved to: {map_path}")
    plt.close()
    
    # 3. Precision and Recall Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['metrics/precision(B)'], label='Precision', linewidth=3, alpha=0.9, color='coral')
    ax.plot(epochs, df['metrics/recall(B)'], label='Recall', linewidth=3, alpha=0.9, color='teal')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision & Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    pr_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    try:
        print(f"✓ Precision/Recall curves saved to: {pr_path}")
    except UnicodeEncodeError:
        print(f"[OK] Precision/Recall curves saved to: {pr_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Training Summary Statistics")
    print(f"{'='*60}")
    print(f"Total Epochs: {len(df)}")
    print(f"\nFinal Metrics:")
    print(f"  mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
    print(f"  mAP50-95: {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
    print(f"  Precision: {df['metrics/precision(B)'].iloc[-1]:.4f}")
    print(f"  Recall: {df['metrics/recall(B)'].iloc[-1]:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Best mAP50: {df['metrics/mAP50(B)'].max():.4f} (Epoch {df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']:.0f})")
    print(f"  Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f} (Epoch {df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']:.0f})")
    print(f"  Best Precision: {df['metrics/precision(B)'].max():.4f} (Epoch {df.loc[df['metrics/precision(B)'].idxmax(), 'epoch']:.0f})")
    print(f"  Best Recall: {df['metrics/recall(B)'].max():.4f} (Epoch {df.loc[df['metrics/recall(B)'].idxmax(), 'epoch']:.0f})")
    print(f"{'='*60}\n")


def main():
    """Main function for plotting training results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training plots from YOLO results CSV')
    parser.add_argument('--results', type=str, default='outputs/train/results.csv',
                       help='Path to results.csv file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: same as CSV directory)')
    
    args = parser.parse_args()
    
    # If running from scripts directory, adjust paths
    if not os.path.exists(args.results):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(project_root, args.results)
        if os.path.exists(results_path):
            args.results = results_path
            if args.output is None:
                args.output = os.path.join(project_root, 'outputs', 'train')
        else:
            print(f"Error: Results file not found: {args.results}")
            sys.exit(1)
    
    plot_training_curves(args.results, args.output)


if __name__ == '__main__':
    main()

