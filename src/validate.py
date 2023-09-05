from operator import itemgetter
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader

from data_utils import ShoesDataset, FashionIQDataset,CIRRDataset
from utils import collate_fn, device


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float]:

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(clip_model: CLIP, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in relative_val_loader:  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)
            if isinstance(batch_predicted_features,tuple):
                batch_predicted_features=batch_predicted_features[0]+batch_predicted_features[1]+batch_predicted_features[2]

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def compute_shoes_val_metrics(relative_val_dataset:ShoesDataset,clip_model:CLIP,index_features:torch.tensor,
                                index_names:List[str],combining_function:callable)->Tuple[float,float]:
    # Generate predictions
    predicted_features, target_names, reference_names = generate_shoes_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T

    for i in range(len(reference_names)):
        ref_name = reference_names[i]
        index = index_names.index(ref_name)
        distances[i][index] = float('inf')

    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at1, recall_at10, recall_at50

def generate_shoes_val_predictions(clip_model:CLIP,relative_val_dataset:ShoesDataset,
                                    combining_function:callable,index_names: List[str], index_features: torch.tensor)->\
                                    Tuple[torch.tensor,List[str]]:
    relative_val_loader=DataLoader(dataset=relative_val_dataset,batch_size=32,num_workers=4,
        pin_memory=True,collate_fn=collate_fn)
    name_to_feat=dict(zip(index_names,index_features))
    predicted_features=torch.empty((0,clip_model.visual.output_dim)).to(device,non_blocking=True)
    target_names=[]
    final_reference_names=[]
    for reference_names,batch_target_names,captions in relative_val_loader:
        text_inputs=clip.tokenize(captions,context_length=77).to(device, non_blocking=True)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)
            if isinstance(batch_predicted_features,tuple):
                batch_predicted_features=batch_predicted_features[0]+batch_predicted_features[1]+batch_predicted_features[2]
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        target_names.extend(batch_target_names)
        final_reference_names.extend(reference_names)
    return predicted_features,target_names,final_reference_names


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in relative_val_loader:  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)
            if isinstance(batch_predicted_features,tuple):
                batch_predicted_features=batch_predicted_features[0]+batch_predicted_features[1]+batch_predicted_features[2]

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members
