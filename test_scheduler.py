import urllib.request
import urllib.parse
import json
import time

BASE_URL = "http://localhost:8000/api"
SESSION_ID = "test_web_app_session_003"
NAMESPACE = "eduwhistle"

def send_message(msg):
    print(f"\nUser: {msg}")
    url = f"{BASE_URL}/chat"
    data = json.dumps({
        "session_id": SESSION_ID,
        "message": msg,
        "namespace": NAMESPACE
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            print(f"Bot: {res_json.get('reply', 'No reply')}")
    except Exception as e:
        print(f"Error: {e}")

print("Starting conversation...")
send_message("Hi, I want to create a new web application for my online tutoring business.")
time.sleep(1)
send_message("The platform needs to allow students to book sessions with tutors, manage schedules, and process payments.")
time.sleep(1)
send_message("My name is John Doe, and my email is john.doe@example.com. My company is TutorMaster.")
time.sleep(1)
send_message("I'd like to use modern tech like React for frontend and Node.js for backend.")
time.sleep(1)
send_message("Can we get started on the planning phase soon? Please contact me back.")

print("\nConversation complete. Triggering daily summary scheduler...")
trigger_url = f"{BASE_URL}/admin/trigger-daily-summary"
req = urllib.request.Request(trigger_url, method="POST")
try:
    with urllib.request.urlopen(req) as response:
        res_body = response.read().decode("utf-8")
        res_json = json.loads(res_body)
        print(f"Trigger response: {res_json}")
except Exception as e:
    print(f"Error triggering summary: {e}")
