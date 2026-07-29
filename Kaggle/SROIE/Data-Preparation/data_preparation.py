"""
SROIE2019 data preparation pipeline.

Reconstructed from Data-preparation.ipynb (screenshots). Turns raw OCR box files
+ entity JSONs into per-document graph-connection CSVs with IOB-style entity
labels, ready for a graph-conv / GAT style layout model.

Pipeline:
  1. Parse raw `*.txt` box files (8 coords + text per line) -> per-doc CSV.
  2. form_graph_connection: for every text box, find nearest neighbour
     directly below (vertical) and directly to the right (horizontal),
     dedup so each node keeps only its single nearest parent.
  3. assign_labels: fuzzy-match each line of OCR text against the ground
     truth entity JSON (company/date/address/total) and tag it.
"""
import os
import json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SROIE_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SROIE2019"
)

SPLITS = ["train", "test"]


# --------------------------------------------------------------------------- #
# Step 1: parse raw OCR box files into per-document CSVs
# --------------------------------------------------------------------------- #
def parse_box_files(split: str):
    box_path = os.path.join(SROIE_FOLDER_PATH, split, "box")
    df_save_path = os.path.join(SROIE_FOLDER_PATH, split, "df")
    os.makedirs(df_save_path, exist_ok=True)

    all_files = os.listdir(box_path)
    for file in tqdm(all_files, desc=f"parsing box files [{split}]"):
        bbox_and_words_list = []
        bbox_file_path = os.path.join(box_path, file)
        with open(bbox_file_path, "r", errors="ignore") as f:
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue

                split_lines = line.split(",")

                bbox = np.array(split_lines[0:8], dtype=np.int32)
                text = str(",".join(split_lines[8:]))
                bbox_and_words_list.append([*bbox, text])

        df = pd.DataFrame(
            bbox_and_words_list,
            columns=["x_min", "y_min", "x1", "y1", "x_max", "y_max", "x3", "y3", "text"],
        )
        df.drop(columns=["x1", "y1", "x3", "y3"], inplace=True)
        df.index.name = "index"
        df.to_csv(os.path.join(df_save_path, file[:-4] + ".csv"))


# --------------------------------------------------------------------------- #
# Step 2: entity ground-truth loading + line-level label assignment
# --------------------------------------------------------------------------- #
def read_entities(path):
    with open(path, "r") as f:
        data = json.load(f)

    dataframe = pd.DataFrame([data])
    return dataframe


def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    for i, column in enumerate(entities):
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()

        matches_count = 0
        for l in line_set:
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                matches_count += 1

        if (column.upper() == "ADDRESS" and (matches_count / len(line_set)) >= 0.5) or \
           (column.upper() != "ADDRESS" and (matches_count == len(line_set))) or \
           matches_count == len(entity_set):
            return column.upper()

    return "O"


def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # value, index
    already_labeled = {
        "TOTAL": False,
        "DATE": False,
        "ADDRESS": False,
        "COMPANY": False,
        "O": False,
    }

    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words["text"]):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if (label == "ADDRESS" and already_labeled["TOTAL"]) or \
           (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
            label = "O"

        # Assign to the largest bounding box
        if label in ["TOTAL", "DATE"]:
            x0_loc = words.columns.get_loc("x_min")
            bbox = words.iloc[i, x0_loc:x0_loc + 4].to_list()
            area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

            if max_area[label][0] < area:
                max_area[label] = (area, i)

            label = "O"

        labels.append(label)

    labels[max_area["DATE"][1]] = "DATE"
    labels[max_area["TOTAL"][1]] = "TOTAL"

    words["label"] = labels
    return words


# --------------------------------------------------------------------------- #
# Step 3: graph connections (nearest neighbour below / to the right)
# --------------------------------------------------------------------------- #
def form_graph_connection(df: pd.DataFrame):
    df_plot = pd.DataFrame()

    # =================== vertical =================== #
    distances, nearest_dest_ids_vert = [], []
    x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, y_dest_coords_vert = [], [], [], []

    # =================== horizontal =================== #
    lengths, nearest_dest_ids_hori = [], []
    x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, y_dest_coords_hori = [], [], [], []

    for src_idx, src_row in df.iterrows():
        # =================== vertical =================== #
        src_range_x = (src_row["x_min"], src_row["x_max"])
        src_center_y = (src_row["y_min"] + src_row["y_max"]) / 2

        dest_attr_vert = []

        # =================== horizontal =================== #
        src_range_y = (src_row["y_min"], src_row["y_max"])
        src_center_x = (src_row["x_min"] + src_row["x_max"]) / 2

        dest_attr_hori = []

        ############## iterate over destination objects ##############
        for dest_idx, dest_row in df.iterrows():
            # flag to signal whether the destination object is below source
            is_beneath = False
            if not src_idx == dest_idx:
                # ======================== vertical ========================#
                dest_range_x = (dest_row["x_min"], dest_row["x_max"])
                dest_center_y = (dest_row["y_min"] + dest_row["y_max"]) / 2

                height = dest_center_y - src_center_y

                # consider only the cases where destination object lies below source
                if dest_center_y > src_center_y:
                    # check if horizontal range of dest lies within range of source

                    # case 1
                    if dest_range_x[0] <= src_range_x[0] and \
                       dest_range_x[1] >= src_range_x[1]:
                        x_common = (src_range_x[0] + src_range_x[1]) / 2

                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True

                    # case 2
                    elif dest_range_x[0] >= src_range_x[0] and \
                         dest_range_x[1] <= src_range_x[1]:
                        x_common = (dest_range_x[0] + dest_range_x[1]) / 2

                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True

                    # case 3
                    elif dest_range_x[0] <= src_range_x[0] and \
                         dest_range_x[1] >= src_range_x[0] and \
                         dest_range_x[1] < src_range_x[1]:
                        x_common = (src_range_x[0] + dest_range_x[1]) / 2

                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True

                    # case 4
                    elif dest_range_x[0] <= src_range_x[1] and \
                         dest_range_x[1] >= src_range_x[1] and \
                         dest_range_x[0] > src_range_x[0]:
                        x_common = (dest_range_x[0] + src_range_x[1]) / 2

                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True

                if not is_beneath:
                    # ========================= horizontal =====================#
                    dest_range_y = (dest_row["y_min"], dest_row["y_max"])
                    # get center of destination NOTE: not used
                    dest_center_x = (dest_row["x_min"] + dest_row["x_max"]) / 2

                    # get length from destination center to source center
                    if dest_center_x > src_center_x:
                        length = dest_center_x - src_center_x
                    else:
                        length = 0

                    # consider only the cases where the destination object
                    # lies to the right of source
                    if dest_center_x > src_center_x:
                        # check if vertical range of dest lies within range
                        # of source

                        # case 1
                        if dest_range_y[0] >= src_range_y[0] and \
                           dest_range_y[1] <= src_range_y[1]:
                            y_common = (dest_range_y[0] + dest_range_y[1]) / 2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 2
                        if dest_range_y[0] <= src_range_y[0] and \
                           dest_range_y[1] <= src_range_y[1] and \
                           dest_range_y[1] > src_range_y[0]:
                            y_common = (src_range_y[0] + dest_range_y[1]) / 2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 3
                        if dest_range_y[0] >= src_range_y[0] and \
                           dest_range_y[1] >= src_range_y[1] and \
                           dest_range_y[0] < src_range_y[1]:
                            y_common = (dest_range_y[0] + src_range_y[1]) / 2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 4
                        if dest_range_y[0] <= src_range_y[0] \
                           and dest_range_y[1] >= src_range_y[1]:
                            y_common = (src_range_y[0] + src_range_y[1]) / 2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

        # sort list of destination attributes by height/length at position
        # 3 in tuple
        dest_attr_vert_sorted = sorted(dest_attr_vert, key=lambda x: x[3])
        dest_attr_hori_sorted = sorted(dest_attr_hori, key=lambda x: x[3])

        # append the index and source and destination coords to draw line
        # ===================== vertical ========================= #
        if len(dest_attr_vert_sorted) == 0:
            nearest_dest_ids_vert.append(-1)
            x_src_coords_vert.append(-1)
            y_src_coords_vert.append(-1)
            x_dest_coords_vert.append(-1)
            y_dest_coords_vert.append(-1)
            distances.append(0)
        else:
            nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
            x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
            y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
            x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
            y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
            distances.append(dest_attr_vert_sorted[0][3])

        # ============================ horizontal ========================= #
        if len(dest_attr_hori_sorted) == 0:
            nearest_dest_ids_hori.append(-1)
            x_src_coords_hori.append(-1)
            y_src_coords_hori.append(-1)
            x_dest_coords_hori.append(-1)
            y_dest_coords_hori.append(-1)
            lengths.append(0)
        else:
            # try and except for the cases where there are vertical
            # connections still to be made but all horizontal connections
            # are accounted for
            try:
                nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
            except IndexError:
                nearest_dest_ids_hori.append(-1)

            try:
                x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
            except IndexError:
                x_src_coords_hori.append(-1)

            try:
                y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
            except IndexError:
                y_src_coords_hori.append(-1)

            try:
                x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
            except IndexError:
                x_dest_coords_hori.append(-1)

            try:
                y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
            except IndexError:
                y_dest_coords_hori.append(-1)

            try:
                lengths.append(dest_attr_hori_sorted[0][3])
            except IndexError:
                lengths.append(0)

    # ===================== vertical ===================== #
    # create df for plotting lines
    df["below_object"] = df[df["index"].isin(nearest_dest_ids_vert)]["text"].values

    # add distances column
    df["below_dist"] = distances

    # add column containing index of destination object
    df["below_obj_index"] = nearest_dest_ids_vert

    # add coordinates for plotting
    df_plot["x_src_vert"] = x_src_coords_vert
    df_plot["y_src_vert"] = y_src_coords_vert
    df_plot["x_dest_vert"] = x_dest_coords_vert
    df_plot["y_dest_vert"] = y_dest_coords_vert

    # ===================== horizontal ===================== #
    # create df for plotting lines
    df["side_object"] = df[df["index"].isin(nearest_dest_ids_hori)]["text"].values

    # add lengths column
    df["side_length"] = lengths

    # add column containing index of destination object
    df["side_obj_index"] = nearest_dest_ids_hori

    # add coordinates for plotting
    df_plot["x_src_hori"] = x_src_coords_hori
    df_plot["y_src_hori"] = y_src_coords_hori
    df_plot["x_dest_hori"] = x_dest_coords_hori
    df_plot["y_dest_hori"] = y_dest_coords_hori

    ########################## concat df and df_plot ##########################
    df_merged = pd.concat([df, df_plot], axis=1)

    # if an object has more than one parent above it, only the connection
    # with the smallest distance is retained and the other distances are
    # replaced by '-1' to get such objects, group by 'below_object' column
    # and use minimum of 'below_dist'

    # ========================= vertical ========================= #
    groups_vert = df_merged.groupby("below_obj_index")["below_dist"].min()
    # groups.index gives a list of the below_object text and groups.values
    # gives the corresponding minimum distance
    groups_dict_vert = dict(zip(groups_vert.index, groups_vert.values))

    # ========================= horizontal ========================= #
    groups_hori = df_merged.groupby("side_obj_index")["side_length"].min()
    # groups.index gives a list of the below_object text and groups.values
    # gives the corresponding minimum distance
    groups_dict_hori = dict(zip(groups_hori.index, groups_hori.values))

    revised_distances_vert = []
    revised_distances_hori = []

    rev_x_src_vert, rev_y_src_vert, rev_x_dest_vert, rev_y_dest_vert = [], [], [], []
    rev_x_src_hori, rev_y_src_hori, rev_x_dest_hori, rev_y_dest_hori = [], [], [], []

    for idx, row in df_merged.iterrows():
        below_idx = row["below_obj_index"]
        side_idx = row["side_obj_index"]

        # ========================= vertical ========================= #
        if row["below_dist"] > groups_dict_vert[below_idx]:
            revised_distances_vert.append(-1)
            rev_x_src_vert.append(-1)
            rev_y_src_vert.append(-1)
            rev_x_dest_vert.append(-1)
            rev_y_dest_vert.append(-1)
        else:
            revised_distances_vert.append(row["below_dist"])
            rev_x_src_vert.append(row["x_src_vert"])
            rev_y_src_vert.append(row["y_src_vert"])
            rev_x_dest_vert.append(row["x_dest_vert"])
            rev_y_dest_vert.append(row["y_dest_vert"])

        # ========================= horizontal ========================= #
        if row["side_length"] > groups_dict_hori[side_idx]:
            revised_distances_hori.append(-1)
            rev_x_src_hori.append(-1)
            rev_y_src_hori.append(-1)
            rev_x_dest_hori.append(-1)
            rev_y_dest_hori.append(-1)
        else:
            revised_distances_hori.append(row["side_length"])
            rev_x_src_hori.append(row["x_src_hori"])
            rev_y_src_hori.append(row["y_src_hori"])
            rev_x_dest_hori.append(row["x_dest_hori"])
            rev_y_dest_hori.append(row["y_dest_hori"])

    # store in dataframe
    # ========================= vertical ========================= #
    df_merged["revised_distances_vert"] = revised_distances_vert
    df_merged["x_src_vert"] = rev_x_src_vert
    df_merged["y_src_vert"] = rev_y_src_vert
    df_merged["x_dest_vert"] = rev_x_dest_vert
    df_merged["y_dest_vert"] = rev_y_dest_vert

    # ========================= horizontal ========================= #
    df_merged["revised_distances_hori"] = revised_distances_hori
    df_merged["x_src_hori"] = rev_x_src_hori
    df_merged["y_src_hori"] = rev_y_src_hori
    df_merged["x_dest_hori"] = rev_x_dest_hori
    df_merged["y_dest_hori"] = rev_y_dest_hori

    return df_merged


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_graphs(split: str):
    df_path = os.path.join(SROIE_FOLDER_PATH, split, "df")
    graph_path = os.path.join(SROIE_FOLDER_PATH, split, "graph")
    os.makedirs(graph_path, exist_ok=True)

    for file in tqdm(os.listdir(df_path), desc=f"building graphs [{split}]"):
        df = pd.read_csv(os.path.join(df_path, file))
        graph_df = form_graph_connection(df)
        graph_df.to_csv(os.path.join(graph_path, file))


def build_annotations(split: str):
    graph_path = os.path.join(SROIE_FOLDER_PATH, split, "graph")
    entities_path = os.path.join(SROIE_FOLDER_PATH, split, "entities")
    annotation_path = os.path.join(SROIE_FOLDER_PATH, split, "annotation")
    os.makedirs(annotation_path, exist_ok=True)

    for file in tqdm(os.listdir(graph_path), desc=f"annotating [{split}]"):
        filename = file[:-4]
        entity_file_path = os.path.join(entities_path, filename + ".txt")
        entity_df = read_entities(entity_file_path)

        bbox_df = pd.read_csv(os.path.join(graph_path, file))

        annotated_df = assign_labels(bbox_df, entity_df)
        annotated_df.to_csv(os.path.join(annotation_path, filename + ".csv"))


def main():
    for split in SPLITS:
        parse_box_files(split)

    for split in SPLITS:
        build_graphs(split)

    for split in SPLITS:
        build_annotations(split)


if __name__ == "__main__":
    main()
