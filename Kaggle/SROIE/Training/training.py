"""
GAT-NER-DGL: Graph Attention Network + BERT + CRF for SROIE key-info extraction.

Reconstructed from GAT-NER-DGL.ipynb (screenshots, Training/ folder). Consumes
the per-document "fet" (feature) Excel files produced downstream of the
Data-Preparation stage -- one row per OCR text box, with its bbox, its
nearest below/side neighbour indices, and its entity tag -- and trains a
3-layer Edge-GAT + linear-CRF model to tag each text box as
company / date / address / total (here: org / O, per the tag_dict below).

NOTE on fidelity: the notebook's `train()` function body was cut off in the
source photos right after the batch loop opens (`for iter, (input_graph,
input_tags) in enumerate(train_loader):`). Everything up to that line is a
direct transcription. The loop body below (forward pass, CRF loss, backward,
optimizer step, epoch logging, early stopping using best_acc /
best_components_error / patience / epochs_no_improvement) is a
reconstruction consistent with the variables the notebook had already
declared -- not a transcription. It's flagged inline with "RECONSTRUCTED".
"""
import math
import os

import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import dgl
from dgl.nn.pytorch.conv import EGATConv
from torch.optim.lr_scheduler import StepLR
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
EMBEDDING_SHAPE = 768
BATCH_SIZE = 32
NODE_TEXT_MAX_LENGTH = 10
EPOCH = 10


# --------------------------------------------------------------------------- #
# Model Architecture
# --------------------------------------------------------------------------- #
class GATLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim,
                 edge_output_dim, multihead_num=1, dropout=0.1):
        super(GATLayer, self).__init__()

        self.conv1 = EGATConv(input_dim, edge_input_dim, hidden_dim, edge_output_dim, multihead_num)
        self.conv2 = EGATConv(hidden_dim * multihead_num, edge_output_dim * multihead_num,
                               hidden_dim, edge_output_dim, multihead_num)
        self.conv3 = EGATConv(hidden_dim * multihead_num, edge_output_dim * multihead_num,
                               hidden_dim, edge_output_dim, multihead_num)

        self.dropout = torch.nn.Dropout(dropout, inplace=True)

    def forward(self, g):
        nfeats = torch.cat([g.ndata["coordinates"], g.ndata["shape"]], dim=1)
        efeats = torch.tensor([g.edata["etype"], g.edata["x_dist"], g.edata["y_dist"],
                                g.edata["relative_height"]])

        h, e = self.conv1(g, nfeats, efeats)
        h = h.view(h.shape[0], -1)
        e = e.view(e.shape[0], -1)

        self.dropout(h)
        self.dropout(e)

        h, e = self.conv2(g, h, e)
        h = h.view(h.shape[0], -1)
        e = e.view(e.shape[0], -1)

        self.dropout(h)
        self.dropout(e)

        h, e = self.conv3(g, h, e)
        h = h.view(h.shape[0], -1)

        return h


class GATBertCRF(torch.nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, output_dim, edge_input_dim,
                 edge_output_dim, multihead_num=1):
        super(GATBertCRF, self).__init__()

        self.gat = GATLayer(input_dim, hidden_dim, output_dim, edge_input_dim,
                             edge_output_dim, multihead_num)
        self.mlp = torch.nn.Linear(output_dim + EMBEDDING_SHAPE, num_classes)
        self.crf = CRF(num_classes)
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, g, train=False):
        h = self.gat(g)

        emissions = torch.zeros([self.input_dim, EMBEDDING_SHAPE, self.num_classes])

        for i in range(0, g.nodes().shape[0]):
            h_i = h[i]
            embeddings_i = g.ndata["embeddings"][i]
            for j in range(0, embeddings_i.shape[0]):
                h_ij = torch.cat((h_i, embeddings_i[j]))
                h_ij = self.mlp(h_ij)
                emissions[i][j] = h_ij

        emissions = emissions.permute(1, 0, 2)

        # For training model
        if train:
            tags = g.ndata["tags"]
            if tags is not None:
                print(f"Emission shape: {emissions.shape}")
                print(f"Tags shape: {tags.T.shape}")
                log_likelihood, sequence_of_tags = self.crf(emissions, tags.T), self.crf.decode(emissions)
                return -1 * log_likelihood, sequence_of_tags

        # For prediction
        sequence_of_tags = self.crf.decode(emissions)
        return sequence_of_tags


# --------------------------------------------------------------------------- #
# Data Preparation
# --------------------------------------------------------------------------- #
def tag_to_ix(tag):
    tag_dict = {
        "B-org": 0,
        "I-org": 1,
        "o": 2,
    }

    return tag_dict[tag]


def direction_to_idx(d):
    dir_dict = {
        "right": 0,
        "left": 1,
        "top": 2,
        "bottom": 3,
    }

    return dir_dict[d]


class GraphDataset(data.Dataset):
    def __init__(self, tokenizer, bert, sample_dir_path):
        self.bert = bert
        self.tokenizer = tokenizer
        self.sample_dir_path = sample_dir_path
        self.files = os.listdir(sample_dir_path)

    def __getitem__(self, index):
        g, tags = self.create_graph(self.sample_dir_path + self.files[index])
        return g, tags

    def __len__(self):
        return len(self.files)

    def create_graph(self, file):
        # read fet file as df
        fet_df = pd.read_excel(file, engine="openpyxl")
        # Dimensions of the document
        width = min(fet_df["x_max"]) - max(fet_df["x_min"])
        height = min(fet_df["y_max"]) - max(fet_df["y_min"])
        # Each boundingbox - dimensions
        fet_df["height"] = fet_df["y_max"] - fet_df["y_min"]
        fet_df["width"] = fet_df["x_max"] - fet_df["x_min"]
        # intializicng the graph
        G = nx.DiGraph()
        # Initialize tags
        tags = []

        # adding the node properties
        for i in range(0, len(fet_df)):
            if "pseudo_id" not in fet_df.columns:
                idx = fet_df[fet_df.columns[0]][i]
            else:
                idx = fet_df["pseudo_id"][i]

            # Bert embedding
            text = fet_df["text"][i]
            encoded_texts = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=NODE_TEXT_MAX_LENGTH,
                truncation=True,
            )
            embeddings = self.bert(**encoded_texts)[0]
            # Add Node to Graph
            G.add_node(
                int(idx),
                embedding=embeddings,
                coordinates=[fet_df["x_min"][i], fet_df["x_max"][i], fet_df["y_min"][i], fet_df["y_max"][i]],
                shape=[fet_df["y_max"][i] - fet_df["y_min"][i], fet_df["x_max"][i] - fet_df["x_min"][i]],
            )
            tags.append([tag_to_ix(t) for t in fet_df["tag"][i].split(",")])

        # adding the edge properties
        for i in range(0, len(fet_df)):
            if "pseudo_id" not in fet_df.columns:
                idx = fet_df[fet_df.columns[0]][i]
            else:
                idx = fet_df["pseudo_id"][i]

            # check if side node exists for a node
            if fet_df["side_node_idx"][i] != -1 and math.isnan(fet_df["side_node_idx"][i]) == False:
                # for every node-side node pair, create two edges with type as right and left
                G.add_edge(
                    int(idx),
                    int(fet_df["side_node_idx"][i]),
                    etype=direction_to_idx("right"),
                    x_dist=abs(int(G.nodes[idx]["x_min"]) - int(G.nodes[fet_df["side_node_idx"][i]]["x_min"])) / width,
                    y_dist=abs(int(G.nodes[idx]["y_min"]) - int(G.nodes[fet_df["side_node_idx"][i]]["y_min"])) / height,
                    relative_height=G.nodes[idx]["height"] / G.nodes[fet_df["side_node_idx"][i]]["height"],
                    relative_width=G.nodes[idx]["width"] / G.nodes[fet_df["side_node_idx"][i]]["width"],
                )

                G.add_edge(
                    int(fet_df["side_node_idx"][i]),
                    int(idx),
                    etype=direction_to_idx("left"),
                    x_dist=abs(int(G.nodes[idx]["x_min"]) - int(G.nodes[fet_df["side_node_idx"][i]]["x_min"])) / width,
                    y_dist=abs(int(G.nodes[idx]["y_min"]) - int(G.nodes[fet_df["side_node_idx"][i]]["y_min"])) / height,
                    relative_height=G.nodes[fet_df["side_node_idx"][i]]["height"] / G.nodes[idx]["height"],
                    relative_width=G.nodes[fet_df["side_node_idx"][i]]["width"] / G.nodes[idx]["width"],
                )

            # check if below node exists for a node
            if fet_df["below_node_idx"][i] != -1 and math.isnan(fet_df["below_node_idx"][i]) == False:
                # for every node-below node pair, create two edges with type as top and below
                G.add_edge(
                    int(idx),
                    int(fet_df["below_node_idx"][i]),
                    etype=direction_to_idx("bottom"),
                    x_dist=abs(int(G.nodes[idx]["x_min"]) - int(G.nodes[fet_df["below_node_idx"][i]]["x_min"])) / width,
                    y_dist=abs(int(G.nodes[idx]["y_min"]) - int(G.nodes[fet_df["below_node_idx"][i]]["y_min"])) / height,
                    relative_height=G.nodes[idx]["height"] / G.nodes[fet_df["below_node_idx"][i]]["height"],
                    relative_width=G.nodes[idx]["width"] / G.nodes[fet_df["below_node_idx"][i]]["width"],
                )

                G.add_edge(
                    int(fet_df["below_node_idx"][i]),
                    int(idx),
                    etype=direction_to_idx("top"),
                    x_dist=abs(int(G.nodes[idx]["x_min"]) - int(G.nodes[fet_df["below_node_idx"][i]]["x_min"])) / width,
                    y_dist=abs(int(G.nodes[idx]["y_min"]) - int(G.nodes[fet_df["below_node_idx"][i]]["y_min"])) / height,
                    relative_height=G.nodes[fet_df["below_node_idx"][i]]["height"] / G.nodes[idx]["height"],
                    relative_width=G.nodes[fet_df["below_node_idx"][i]]["width"] / G.nodes[idx]["width"],
                )

        # Convert to DGL graph
        dgl_graph = dgl.from_networkx(
            G,
            node_attrs=["embedding", "coordinates", "shape"],
            edge_attrs=["etype", "relative_height", "relative_width"],
        )

        return dgl_graph, tags


# --------------------------------------------------------------------------- #
# Load Data
# --------------------------------------------------------------------------- #
def collate(samples):
    graphs, tags = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    tags = dgl.batch(tags)
    return batched_graph, tags


train_dir = "Data/training_data/"
test_dir = "Data/testing_data/"

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert = BertModel.from_pretrained("bert-base-cased")

trainset = GraphDataset(tokenizer, bert, train_dir)
testset = GraphDataset(tokenizer, bert, test_dir)

train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=collate)


# --------------------------------------------------------------------------- #
# Model Training
# --------------------------------------------------------------------------- #
model = GATBertCRF(
    num_classes=3,
    input_dim=EMBEDDING_SHAPE + 4,  # coordinates (4) concatenated with shape (2) per GATLayer.forward -> adjust to match nfeats width
    hidden_dim=256,
    output_dim=256,
    edge_input_dim=4,
    edge_output_dim=32,
    multihead_num=4,
)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, 5, gamma=0.9)


def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    train_log = open("train_log.txt", "w")
    train_log.close()

    best_acc = 0
    best_components_error = 200
    patience = 100
    epochs_no_improvement = 0

    for epoch in range(EPOCH):
        epoch_loss = 0
        print("\n\n")
        model.training = True
        for iter, (input_graph, input_tags) in enumerate(train_loader):
            # ------------------------------------------------------------- #
            # RECONSTRUCTED from here down -- not visible in source photos.
            # Batch loop body inferred from the variables already declared
            # above (optimizer, scheduler, best_acc, best_components_error,
            # patience, epochs_no_improvement, train_log) and from
            # GATBertCRF.forward's train=True contract (returns
            # negative log-likelihood loss + decoded tag sequence).
            # ------------------------------------------------------------- #
            if torch.cuda.is_available():
                input_graph = input_graph.to("cuda")
                input_tags = input_tags.to("cuda")

            input_graph.ndata["tags"] = input_tags

            optimizer.zero_grad()
            loss, _ = model(input_graph, train=True)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            with open("train_log.txt", "a") as train_log:
                train_log.write(f"epoch {epoch} iter {iter} loss {loss.item():.4f}\n")

        scheduler.step()
        epoch_loss /= max(iter + 1, 1)
        print(f"Epoch {epoch} | mean loss {epoch_loss:.4f}")

        # --- validation / early stopping (RECONSTRUCTED) --- #
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_graph, input_tags in test_loader:
                if torch.cuda.is_available():
                    input_graph = input_graph.to("cuda")
                    input_tags = input_tags.to("cuda")
                predicted_tags = model(input_graph, train=False)
                predicted_tags = torch.tensor(predicted_tags)
                correct += (predicted_tags == input_tags.cpu()).sum().item()
                total += input_tags.numel()
        model.train()

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch} | val acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model


if __name__ == "__main__":
    train(model)
