"""
Example script demonstrating multi-GPU training with TrialBot.
This example shows how to use DataParallel, DistributedDataParallel, and DeepSpeed.
"""

import torch
import torch.nn as nn
import argparse
from trialbot.training.trial_bot import TrialBot
from trialbot.data.ns_vocabulary import NSVocabulary


# Define a simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids).mean(dim=1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(), labels.float())
            output['loss'] = loss
        
        return output


def get_model(hparams, vocab):
    """Model creation function required by TrialBot."""
    vocab_size = len(vocab.get_namespace('tokens')) if vocab else 1000
    return SimpleModel(vocab_size=vocab_size)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Training Example')
    
    # TrialBot arguments
    parser.add_argument('--dataset', default='demo', help='Dataset name')
    parser.add_argument('--hparamset', default='demo', help='Hyperparameter set')
    parser.add_argument('--translator', default='demo', help='Translator name')
    
    # Multi-GPU arguments
    parser.add_argument('--gpus', type=str, default='0,1', 
                       help='GPU IDs to use (comma-separated), e.g., "0,1,2,3"')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                       help='Use DistributedDataParallel instead of DataParallel')
    parser.add_argument('--deepspeed', action='store_true',
                       help='Enable DeepSpeed training')
    parser.add_argument('--deepspeed-config', type=str, default=None,
                       help='Path to DeepSpeed configuration file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-GPU Training Example with TrialBot")
    print("=" * 60)
    print(f"GPUs: {args.gpus}")
    print(f"Distributed: {args.multiprocessing_distributed}")
    print(f"DeepSpeed: {args.deepspeed}")
    print("=" * 60)
    
    # Create TrialBot instance
    bot = TrialBot(
        args=args,
        trial_name="multi_gpu_demo",
        get_model_func=get_model
    )
    
    # Run training
    print("\nStarting training...")
    bot.run()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
