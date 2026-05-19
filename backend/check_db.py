import os
from pymongo import MongoClient
from dotenv import load_dotenv
import json
from bson import json_util

load_dotenv()

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db = client["chatbot_sessions"]
collection = db["chatbot"]

# Get the latest session
latest_session = collection.find().sort("updated_at", -1).limit(1)

for session in latest_session:
    print(json.dumps(session, default=json_util.default, indent=2))
