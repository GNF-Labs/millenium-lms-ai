from flask import Flask, request, jsonify
from flask_cors import CORS
from coba import get_single_item_recommendation
import traceback
import logging
import torch

app = Flask(__name__)

# Set up CORS to allow requests from any origin
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to get model recommendations
def get_model_recommendations(user_id, item_id):
    try:
        # Ensure that get_single_item_recommendation handles device properly
        recommended_items = get_single_item_recommendation('./data/data_collect.csv', user_id, item_id)
        return recommended_items
    except Exception as e:
        # Log the exception details
        app.logger.error(f"Error in get_model_recommendations: {e}")
        raise

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    item_id = request.args.get('item_id')

    # Validate the presence of user_id and item_id
    if user_id is None or item_id is None:
        return jsonify({'error': 'user_id and item_id are required'}), 400

    # Convert to integers and handle potential errors
    try:
        user_id = int(user_id)
        item_id = int(item_id)
    except ValueError:
        return jsonify({'error': 'user_id and item_id must be integers'}), 400

    # Validate user_id range
    if not (0 <= user_id < 999999):
        return jsonify({'error': f'Invalid user_id {user_id}. Valid range is 0-99998.'}), 404

    # Validate item_id range
    if not (0 <= item_id < 17):
        return jsonify({'error': f'Invalid item_id {item_id}. Valid range is 0-16.'}), 404

    try:
        # Get recommendations
        recommended_items = get_model_recommendations(user_id, item_id)
        return jsonify({
            'user_id': user_id,
            'item_id': item_id,
            'recommended_items': recommended_items.tolist(),
        })
    except Exception as e:
        # Capture detailed exception information
        error_message = f"An error occurred while getting recommendations: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 500

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Hello! Welcome to the recommendation system API.'})

if __name__ == '__main__':
    app.run(debug=True)
