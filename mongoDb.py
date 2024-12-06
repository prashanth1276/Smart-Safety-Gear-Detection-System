from pymongo import MongoClient
from datetime import datetime, timedelta

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI if using a remote instance
db = client['safety_gear']  # Database name
detections = db['detections']  # Collection name

# Initialize the database (optional for MongoDB as it creates collections automatically)
def init_db():
    # Optionally ensure indexes for faster queries (e.g., on file_name or detection_time)
    detections.create_index("detection_time")

# Save detection results to the database with IST timestamp
def save_results(file_name, detected_items, missing_items):
    # Get current UTC time and adjust it to IST (UTC +5:30)
    current_time_utc = datetime.utcnow()
    current_time_ist = current_time_utc + timedelta(hours=5, minutes=30)
    
    # Format the time without microseconds (using strftime)
    current_time_str = current_time_ist.strftime('%Y-%m-%d %H:%M:%S')  # Exclude microseconds

    # If there are no missing items, add a professional message
    if not missing_items:
        missing_items = ["All essential safety gear is detected."]  # Default message when nothing is missing

    # Use a set to get unique detected items
    unique_detected_items = list(set(detected_items))

    # Insert document into MongoDB
    detection_record = {
        "file_name": file_name,
        "detection_time": current_time_str,  # Store as string without microseconds
        "detected_items": unique_detected_items,
        "missing_items": missing_items
    }

    # Assuming `detections` is the MongoDB collection where results are stored
    detections.insert_one(detection_record)

# Fetch all detection records from the database
def get_history():
    # Retrieve all records sorted by detection_time (newest first)
    records = list(detections.find({}, {"_id": 0}).sort("detection_time", -1))
    return records

# Function to clear all detection records if needed
def clear_all_history():
    # Delete all documents from the collection
    detections.delete_many({})

# Log rotation: Archive or delete old records (older than 30 days)
def log_rotation(days_threshold=30):
    # Calculate the cutoff date (30 days ago)
    cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

    # Archive old records (optional: here, we delete records older than threshold)
    # You could also move these to another collection or database if you want to archive instead of delete.
    detections.delete_many({"detection_time": {"$lt": cutoff_date_str}})