import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cv_summary_test():
    """Test the plot_cv_summary function with dummy data"""
    # Dummy data
    final_aucs = [0.85, 0.82, 0.88, 0.79, 0.91]
    final_accuracies = [0.80, 0.78, 0.85, 0.75, 0.88]
    final_val_losses = [0.45, 0.52, 0.38, 0.61, 0.33]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # AUC
    axes[0].bar(range(1, 6), final_aucs, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axhline(np.mean(final_aucs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_aucs):.3f}')
    axes[0].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_aucs) - np.std(final_aucs),
                        np.mean(final_aucs) + np.std(final_aucs), alpha=0.2, color='red')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('Validation AUC per Fold', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].set_xticks(range(1, 6))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Accuracy
    axes[1].bar(range(1, 6), final_accuracies, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].axhline(np.mean(final_accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_accuracies):.3f}')
    axes[1].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_accuracies) - np.std(final_accuracies),
                        np.mean(final_accuracies) + np.std(final_accuracies), alpha=0.2, color='red')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy per Fold', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].set_xticks(range(1, 6))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Val Loss
    axes[2].bar(range(1, 6), final_val_losses, color='coral', edgecolor='black', alpha=0.7)
    axes[2].axhline(np.mean(final_val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_val_losses):.3f}')
    axes[2].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_val_losses) - np.std(final_val_losses),
                        np.mean(final_val_losses) + np.std(final_val_losses), alpha=0.2, color='red')
    axes[2].set_xlabel('Fold', fontsize=12)
    axes[2].set_ylabel('Validation Loss', fontsize=12)
    axes[2].set_title('Validation Loss per Fold', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(1, 6))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test_cv_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Test plot saved successfully!")

if __name__ == "__main__":
    plot_cv_summary_test()