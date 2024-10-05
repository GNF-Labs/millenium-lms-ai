from flask import Flask, request, jsonify
from flask_cors import CORS
from coba import get_single_item_recommendation
from coba2 import train_jodie_model
import traceback
import logging
import torch
import psycopg2
import csv
import os
import logging

app = Flask(__name__)

# Set up CORS to allow requests from any origin
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database configuration
DB_HOST = "cancer.akhdani.net"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASS = "w2e3r4t5"
DB_NAME = "lms"

# Path to save the CSV file
CSV_FILE_PATH = './data/data_collect.csv'

# Function to connect to the PostgreSQL database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        return conn
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        raise

# Function to fetch data from the database and save it to a CSV file
# Function to fetch data from the database, assign interaction_order, and save it to a CSV file
# Function to fetch data from the database, assign interaction_order, and save it to a CSV file
# Function to fetch data from the database and save it to a CSV file
def fetch_and_save_data():
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch the required data from the table 'user_course_interactions'
        query = """
        SELECT
            user_id,
            course_id,
            last_interaction,
            completed,
            registered,
            viewed,
            completion_rate,
            n_completed_chapters,
            n_completed_sub_chapters
        FROM user_course_interactions
        ORDER BY user_id, last_interaction
        """
        cursor.execute(query)
        data = cursor.fetchall()

        # Create a new list to store the updated data with interaction_order
        updated_data = []
        current_user = None
        interaction_order = 0

        # Iterate over the fetched data and assign interaction_order
        for row in data:
            if len(row) != 9:  # Ensure the row has the expected number of columns
                app.logger.error(f"Unexpected number of values in row: {row}")
                continue  # Skip this row or handle the error as needed

            # Unpack all the 9 values
            user_id, course_id, last_interaction, completed, registered, viewed, completion_rate, n_completed_chapters, n_completed_sub_chapters = row

            # Convert boolean fields to integers
            completed = 1 if completed else 0
            registered = 1 if registered else 0
            viewed = 1 if viewed else 0

            # Check if we are still processing the same user
            if user_id != current_user:
                # New user, reset interaction order
                current_user = user_id
                interaction_order = 1
            else:
                # Same user, increment the interaction order
                interaction_order += 1

            # Append the row with the new interaction_order
            updated_data.append([
                user_id, course_id, interaction_order,
                completed, registered, viewed, n_completed_chapters, n_completed_sub_chapters
            ])

        # Get column names
        colnames = [
            'userid_DI_encoded', 'course_id_encoded', 'interaction_order',
            'completed', 'registered', 'viewed', 'n_completed_chapters', 'n_completed_sub_chapters'
        ]

        # Ensure the directory for saving the file exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

        # Write the updated data to CSV
        with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(colnames)
            # Write the data rows
            csvwriter.writerows(updated_data)

        cursor.close()
        conn.close()

        app.logger.info(f"Data successfully saved to {CSV_FILE_PATH}")

    except Exception as e:
        app.logger.error(f"Error fetching data from the database: {e}")
        raise

# API route to trigger data fetching and saving to CSV
@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    try:
        # Fetch and save the data
        fetch_and_save_data()
        return jsonify({'message': f'Data successfully fetched and saved to {CSV_FILE_PATH}'}), 200
    except Exception as e:
        error_message = f"An error occurred while fetching and saving data: {str(e)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 500


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

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()

    # Extract parameters from the JSON payload
    network = data.get('network', 'data_collect')
    model_name = data.get('model_name', 'jodie')
    epochs = data.get('epochs', 1)
    gpu = data.get('gpu', -1)
    embedding_dim = data.get('embedding_dim', 128)
    train_proportion = data.get('train_proportion', 0.8)
    state_change = data.get('state_change', True)

    try:
        # Call the training function
        train_jodie_model(network, model_name, epochs, embedding_dim, train_proportion, state_change)
        return jsonify({"message": "Training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Hello! Welcome to the recommendation system API.'})

if __name__ == '__main__':
    app.run(debug=True)
