import torch
from torch.utils.data import DataLoader, Dataset

# Define constants
SRC_VOCAB_SIZE = 100  # Size of the source vocabulary
TGT_VOCAB_SIZE = 100  # Size of the target vocabulary
SEQ_LENGTH = 10       # Length of each sequence
NUM_SAMPLES = 100     # Number of samples in the dataset
BATCH_SIZE = 16       # Batch size for DataLoader

# Create synthetic vocabularies
src_vocab = {f"token_{i}": i for i in range(SRC_VOCAB_SIZE)}
tgt_vocab = {f"token_{i}": i for i in range(TGT_VOCAB_SIZE)}

# Reverse vocabularies for decoding
src_vocab_reverse = {i: f"token_{i}" for i in range(SRC_VOCAB_SIZE)}
tgt_vocab_reverse = {i: f"token_{i}" for i in range(TGT_VOCAB_SIZE)}

# Synthetic dataset class
class SyntheticTranslationDataset(Dataset):
    def __init__(self, num_samples, seq_length, src_vocab_size, tgt_vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random source and target sequences
        src_seq = torch.randint(2, self.src_vocab_size, (self.seq_length,))
        tgt_seq = torch.randint(2, self.tgt_vocab_size, (self.seq_length,))
        # Add <bos> (beginning of sentence) and <eos> (end of sentence) tokens
        src_seq = torch.cat([torch.tensor([0]), src_seq, torch.tensor([1])])
        tgt_seq = torch.cat([torch.tensor([0]), tgt_seq, torch.tensor([1])])
        return src_seq, tgt_seq

# Create dataset and dataloader
dataset = SyntheticTranslationDataset(NUM_SAMPLES, SEQ_LENGTH, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Example usage
if __name__ == "__main__":
    for src_batch, tgt_batch in data_loader:
        print("Source Batch:", src_batch)
        print("Target Batch:", tgt_batch)
        break  # Only process the first batch for demonstration
