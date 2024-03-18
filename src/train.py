import torch
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def main():
    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")

    device = xm.xla_device()

    train_dataloader = DataLoader(dataset["train"])
    for i in range(10):
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
