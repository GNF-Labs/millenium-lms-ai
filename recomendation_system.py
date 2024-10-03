import torch
import torch.nn as nn
from torch.autograd import Variable

def generate_recommendations(model, user_embeddings, item_embeddings, item_embeddings_static, num_users, num_items, num_features, top_k=10):
    """
    Generate a list of recommended items for each user.

    Parameters:
    - model: Trained JODIE model.
    - user_embeddings: Embeddings for users at the end of training.
    - item_embeddings: Embeddings for items at the end of training.
    - item_embeddings_static: Static embeddings for items.
    - num_users: Total number of users.
    - num_items: Total number of items.
    - num_features: Number of features used in the model.
    - top_k: Number of top recommendations to generate for each user.

    Returns:
    - recommendations: A dictionary where the key is the user_id and the value is a list of recommended item_ids.
    """
    recommendations = {}

    for userid in range(num_users):
        user_item_predictions = []

        for itemid in range(num_items):
            user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
            user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
            item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
            item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]

            feature_tensor = Variable(torch.zeros(1, num_features).cuda())
            user_timediffs_tensor = Variable(torch.zeros(1).cuda())
            item_timediffs_tensor = Variable(torch.zeros(1).cuda())
            item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid])]

            user_projected_embedding = model.forward(
                user_embedding_input, 
                item_embedding_previous, 
                timediffs=user_timediffs_tensor, 
                features=feature_tensor, 
                select='project'
            )

            user_item_embedding = torch.cat([
                user_projected_embedding, 
                item_embedding_previous, 
                item_embeddings_static[torch.cuda.LongTensor([itemid])], 
                user_embedding_static_input
            ], dim=1)

            predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

            euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), 
                                                        torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1)

            top_k_items = torch.topk(-euclidean_distances, top_k).indices.cpu().numpy()

            user_item_predictions.extend(top_k_items)

        # Remove duplicates and limit to top_k items
        user_item_predictions = list(dict.fromkeys(user_item_predictions))[:top_k]

        recommendations[userid] = user_item_predictions

    return recommendations

def load_trained_model():
    # Load the trained JODIE model, embeddings, and other necessary components here.
    # This is just a placeholder function; you'll need to implement the actual loading logic.
    model = None  # Replace with actual model loading
    user_embeddings = None  # Replace with actual embeddings loading
    item_embeddings = None  # Replace with actual embeddings loading
    item_embeddings_static = None  # Replace with actual static embeddings loading
    num_users = None  # Replace with actual number of users
    num_items = None  # Replace with actual number of items
    num_features = None  # Replace with actual number of features

    return model, user_embeddings, item_embeddings, item_embeddings_static, num_users, num_items, num_features

def recommend_for_user(user_id, top_k=10):
    model, user_embeddings, item_embeddings, item_embeddings_static, num_users, num_items, num_features = load_trained_model()
    
    if user_id >= num_users:
        raise ValueError("Invalid user_id")

    recommendations = generate_recommendations(model, user_embeddings, item_embeddings, item_embeddings_static, num_users, num_items, num_features, top_k)

    return recommendations[user_id]

if __name__ == "__main__":
    user_id = 0  # Specify the user ID for whom you want to generate recommendations
    top_k = 10  # Specify how many recommendations you want
    recommendations = recommend_for_user(user_id, top_k)
    print(f"Top {top_k} recommendations for user {user_id}: {recommendations}")
