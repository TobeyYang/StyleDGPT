#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import csv
import json
import math
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange

# from modeling_discriminator import Discriminator
from models import Discriminator

torch.manual_seed(0)
np.random.seed(0)

# example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
example_sentence = 'here we explore one aspect of this non locality as it might appear in _eqn_ yang mills theories . <|endoftext|>'
max_length_seq = 30


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        # data["X"] = self.X[index]
        data["X"] = torch.tensor(self.X[index], dtype=torch.long)
        # seq = torch.tensor(seq[:max_length_seq], device=device, dtype=torch.long)
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def evaluate_performance(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)
        )
    )


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    probs = model(input_t).softmax(dim=-1).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, prob) for c, prob in zip(classes, probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader


def train_discriminator(dataset, dataset_fp=None, model_name_or_path="gpt2-medium",
                        epochs=10, batch_size=64, log_interval=10,
                        save_model=False, cached=False, no_cuda=False, output_dir=None):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset_fp is None:
        raise ValueError("When generic dataset is selected, "
                         "dataset_fp needs to be specified aswell.")

    idx2class = ['non_style', 'style']
    class2idx = {'non_style': 0, 'style': 1}

    discriminator = Discriminator(
        class_size=len(idx2class),
        model_name_or_path=model_name_or_path,
        cached_mode=cached,
        device=device
    ).to(device)
    print(f"Built discriminator with {model_name_or_path}")

    x = []
    y = []
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if i > 2_000_000:
                break

            if row:
                label = row[0]
                text = row[1]

                try:
                    seq = discriminator.tokenizer.encode(text)[:max_length_seq]
                    # seq = torch.tensor(seq[:max_length_seq], device=device, dtype=torch.long)

                    x.append(seq)
                    y.append(class2idx[label])

                except:
                    print("Error tokenizing line {}, skipping it".format(i))
                    pass

    full_dataset = Dataset(x, y)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size]
    )

    discriminator_meta = {
        "class_size": len(idx2class),
        "embed_size": discriminator.embed_size,
        "pretrained_model": model_name_or_path,
        "class_vocab": class2idx,
        "default_class": 0,
    }

    end = time.time()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(test_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    with open(os.path.join(output_dir, "classifier_head_meta.json"), "w") as meta_file:
        json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        evaluate_performance(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class,
                cached=cached, device=device)

        torch.save(discriminator.get_classifier().state_dict(),
                   os.path.join(output_dir, "classifier_head_epoch_{}.pt".format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="arxiv",
                        help="dataset to train the discriminator on."
                             "In case of generic, the dataset is expected"
                             "to be a TSBV file with structure: class \\t text")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=100, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--output_dir", type=str, help='the output directory')
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_discriminator(**(vars(args)))

