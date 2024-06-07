"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.networks import TextEncoder

__all__ = [
    "shuffle_files",
    "nearest_neighbor",
    "get_images_texts",
    "load_or_process_file",
]


def shuffle_files(img_expert_files, txt_expert_files):
    # Check if both lists have the same length and if the lists are not empty
    assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
    assert len(img_expert_files) != 0, "No files to shuffle"
    shuffled_indices = np.random.permutation(len(img_expert_files))

    # Apply the shuffled indices to both lists
    img_expert_files = np.take(img_expert_files, shuffled_indices)
    txt_expert_files = np.take(txt_expert_files, shuffled_indices)
    return img_expert_files, txt_expert_files

def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """Find the nearest neighbors for a batch of embeddings.

    Args:
    sentences: The original sentences from which the embeddings were computed.
    query_embeddings: A batch of embeddings for which to find the nearest neighbors.
    database_embeddings: All pre-computed embeddings.

    Returns:
    A list of the most similar sentences for each embedding in the batch.
    """
    nearest_neighbors = []
    
    for query in query_embeddings:
        similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
        
        most_similar_index = np.argmax(similarities)
        
        nearest_neighbors.append(sentences[most_similar_index])
        
    return nearest_neighbors


def get_images_texts(n, dataset, args, i_have_indices=None):
    """Get random n images and corresponding texts from the dataset.

    Args:
    n: Number of images and texts to retrieve.
    dataset: The dataset containing image-text pairs.

    Returns:
    A tuple containing two elements:
      - A tensor of randomly selected images.
      - A tensor of the corresponding texts, encoded as floats.
    """
    # Generate n unique random indices
    if i_have_indices is not None:
        idx_shuffle = i_have_indices
    else:
        idx_shuffle = np.random.permutation(len(dataset))[:n]

    # Initialize the text encoder
    text_encoder = TextEncoder(args)

    image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
    
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")

    return image_syn, text_syn.float()


def load_or_process_file(file_type, process_func, args, data_source):
    """
    Load the processed file if it exists, otherwise process the data source and create the file.

    Args:
    file_type: The type of the file (e.g., 'train', 'test').
    process_func: The function to process the data source.
    args: The arguments required by the process function and to build the filename.
    data_source: The source data to be processed.

    Returns:
    The loaded data from the file.
    """
    filename = f'{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'


    if not os.path.exists(filename):
        print(f'Creating {filename}')
        process_func(args, data_source)
    else:
        print(f'Loading {filename}')
    
    return np.load(filename)

def get_LC_images_texts(n, dataset, args):
    """Get random n images and corresponding texts from the dataset.

    Args:
    n: Number of images and texts to retrieve.
    dataset: The dataset containing image-text pairs.

    Returns:
    A tuple containing two elements:
      - A tensor of randomly selected images.
      - A tensor of the corresponding texts, encoded as floats.
    """
    # Generate n unique random indices
    idx_shuffle = np.random.permutation(len(dataset))[:n]

    # Initialize the text encoder
    text_encoder = TextEncoder(args)

    image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
    
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")

    return image_syn, text_syn.float()

