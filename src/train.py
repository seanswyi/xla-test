import evaluate
import torch
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)


def main():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = xm.xla_device()

    model = model.to(device)

    epoch_pbar = trange(
        num_epochs,
        desc="Epochs",
        total=num_epochs,
    )
    for epoch in epoch_pbar:
        model.train()

        train_pbar = tqdm(
            iterable=train_dataloader,
            desc="Training",
            total=len(train_dataloader),
        )
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()

        metric = evaluate.load("accuracy")
        eval_pbar = tqdm(
            iterable=test_dataloader,
            desc="Evaluating",
            total=len(test_dataloader),
        )
        for batch in eval_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric.compute()


if __name__ == "__main__":
    main()
