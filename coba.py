import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import logging
from library_data import load_network
from library_models import JODIE, load_model, set_embeddings_training_end
from coba2 import train_jodie_model
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Args class
class Args:
    def __init__(self, network, datapath, embedding_dim, model, gpu, epoch, train_proportion, state_change):
        self.network = network
        self.datapath = datapath
        self.embedding_dim = embedding_dim
        self.model = model
        self.gpu = gpu
        self.epoch = epoch
        self.train_proportion = train_proportion
        self.state_change = state_change

def get_single_item_recommendation(csv_file, user_id, item_id, model_name='jodie', gpu_id=-1, epoch=0,
                                   embedding_dim=128, train_proportion=0.8, state_change=True):
    if train_proportion > 0.8:
        raise ValueError('Training sequence proportion cannot be greater than 0.8.')

    logging.info("Starting recommendation process")

    # Set device to CPU
    device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    network = os.path.splitext(os.path.basename(csv_file))[0]
    if network == "mooc":
        logging.info(f"No interaction prediction for {network}")
        return []

    # Create the args object
    args = Args(network=network, datapath=csv_file, embedding_dim=embedding_dim, model=model_name,
                gpu=gpu_id, epoch=epoch, train_proportion=train_proportion, state_change=state_change)

    # Load the network
    try:
        [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
         item2id, item_sequence_id, item_timediffs_sequence,
         timestamp_sequence, feature_sequence,
         y_true] = load_network(args)
    except Exception as e:
        logging.error(f"Error loading network: {e}")
        raise

    num_users = len(user2id)
    num_items = len(item2id) + 1

    if user_id >= num_users or item_id >= num_items:
        raise ValueError('Invalid user_id or item_id')

    # INITIALIZE MODEL PARAMETERS
    model = JODIE(args, num_features=len(feature_sequence[0]), num_users=num_users, num_items=num_items).to(device)

    # LOAD THE MODEL
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    try:
        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(
            model, optimizer, args, epoch)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # Move embeddings to the CPU
    user_embeddings_dystat = user_embeddings_dystat.to(device)
    item_embeddings_dystat = item_embeddings_dystat.to(device)

    # SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries.to(device),
                                item_embeddings_timeseries.to(device), user_sequence_id, item_sequence_id, train_end_idx_training)

    # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
    item_embeddings = item_embeddings_dystat[:, :embedding_dim].clone()
    item_embeddings_static = item_embeddings_dystat[:, embedding_dim:].clone()
    user_embeddings = user_embeddings_dystat[:, :embedding_dim].clone()
    user_embeddings_static = user_embeddings_dystat[:, embedding_dim:].clone()

    # GET EMBEDDINGS FOR SPECIFIED USER AND ITEM
    user_embedding_input = user_embeddings[torch.LongTensor([user_id]).to(device)]
    user_embedding_static_input = user_embeddings_static[torch.LongTensor([user_id]).to(device)]
    item_embedding_input = item_embeddings[torch.LongTensor([item_id]).to(device)]
    item_embedding_static_input = item_embeddings_static[torch.LongTensor([item_id]).to(device)]

    feature_tensor = Variable(torch.zeros(1, len(feature_sequence[0]))).to(device)
    user_timediffs_tensor = Variable(torch.zeros(1)).to(device)
    item_timediffs_tensor = Variable(torch.zeros(1)).to(device)
    item_embedding_previous = item_embeddings[torch.LongTensor([item_id]).to(device)]

    user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous,
                                             timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous,
                                     item_embeddings_static[torch.LongTensor([item_id]).to(device)],
                                     user_embedding_static_input], dim=1)

    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)
    euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1),
                                                torch.cat([item_embeddings, item_embeddings_static], dim=1).to(device)).squeeze(-1)
    top_10_items = torch.topk(-euclidean_distances, 10).indices.cpu().numpy()

    return top_10_items

def get_prerequisite_course_recommendation(csv_file, target_course_id, model_name='jodie', gpu_id=-1, epoch=0,
                                           embedding_dim=128, train_proportion=0.8, state_change=True):
    if train_proportion > 0.8:
        raise ValueError('Training sequence proportion cannot be greater than 0.8.')

    logging.info("Starting prerequisite course recommendation process")

    # Set device to CPU
    device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    network = os.path.splitext(os.path.basename("./data/data_collect.csv"))[0]
    if network == "mooc":
        logging.info(f"No interaction prediction for {network}")
        return []

    # Create the args object
    args = Args(network=network, datapath="./data/data_collect.csv", embedding_dim=embedding_dim, model=model_name,
                gpu=gpu_id, epoch=epoch, train_proportion=train_proportion, state_change=state_change)

    # Load the network
    try:
        [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
         item2id, item_sequence_id, item_timediffs_sequence,
         timestamp_sequence, feature_sequence,
         y_true] = load_network(args)
    except Exception as e:
        logging.error(f"Error loading network: {e}")
        raise

    num_items = len(item2id) + 1

    if target_course_id >= num_items:
        raise ValueError('Invalid course_id')

    # INITIALIZE MODEL PARAMETERS
    model = JODIE(args, num_features=len(feature_sequence[0]), num_users=len(user2id), num_items=num_items).to(device)

    # LOAD THE MODEL
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    try:
        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(
            model, optimizer, args, epoch)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # Move embeddings to the CPU
    item_embeddings_dystat = item_embeddings_dystat.to(device)

    # SET THE ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
    user_data_id = user_sequence_id  # This is just an example; use appropriate logic to retrieve user data IDs.
    set_embeddings_training_end(user_data_id, item_embeddings_dystat, None, item_embeddings_timeseries.to(device), None, item_sequence_id, train_end_idx_training)


    # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
    item_embeddings = item_embeddings_dystat[:, :embedding_dim].clone()
    item_embeddings_static = item_embeddings_dystat[:, embedding_dim:].clone()

    # GET EMBEDDING FOR THE TARGET COURSE
    target_course_embedding = item_embeddings[torch.LongTensor([target_course_id]).to(device)]
    target_course_static_embedding = item_embeddings_static[torch.LongTensor([target_course_id]).to(device)]

    # Combine dynamic and static embeddings for comparison
    target_course_combined_embedding = torch.cat([target_course_embedding, target_course_static_embedding], dim=1)

    # Compute Euclidean distances to find similar items (potential prerequisites)
    combined_embeddings = torch.cat([item_embeddings, item_embeddings_static], dim=1).to(device)
    euclidean_distances = nn.PairwiseDistance()(target_course_combined_embedding.repeat(num_items, 1), combined_embeddings).squeeze(-1)

    # Find the top 5 closest courses as prerequisites (excluding the target course itself)
    top_5_prerequisites = torch.topk(-euclidean_distances, 6).indices.cpu().numpy()  # 6 to include target, exclude later
    top_5_prerequisites = top_5_prerequisites[top_5_prerequisites != target_course_id][:5]  # Exclude target course

    return top_5_prerequisites

