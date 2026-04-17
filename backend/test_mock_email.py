import asyncio
import sys
import os

# Ensure backend root is in path
sys.path.insert(0, os.path.dirname(__file__))

from models.schemas import Session, ConversationStage
from providers.ollama_provider import OllamaProvider
from services.email_agent import EmailAgent

async def test_send_email():
    print("Testing Email Functionality (Mock Send)...")
    provider = OllamaProvider() # uses defaults http://localhost:11434 and llama3.2
    
    # Let's see if ollama is running
    is_healthy = await provider.health_check()
    if not is_healthy:
        print("Note: Ollama provider health check failed. EmailAgent will use fallback text.")
    else:
        print("Ollama is running. Using LLM for email generation.")
        
    agent = EmailAgent(provider)
    
    # Create a mock session
    session = Session(session_id="test-mock-email-123", stage=ConversationStage.EMAIL)
    session.collected_data.personal_info.name = "Dinesh"
    session.collected_data.personal_info.email = "dinesh.colorwhistle@gmail.com"
    session.collected_data.tech_discovery.project_type = "Web Application"
    session.summary = "Dinesh wants a web application."
    
    try:
        result = await agent.compose_and_send(session)
        
        print("\n--- Test Result ---")
        if result.success:
            print("✅ Email composed and sent successfully (Mocked).")
            print(f"Message: {result.message}")
            if result.user_email:
                print(f"User Email sent to: {result.user_email.to}")
                print(f"User Subject: {result.user_email.subject}")
            else:
                print("No user email was generated. Something might be wrong.")
                
            if result.admin_email:
                print(f"Admin Email sent to: {result.admin_email.to}")
                print(f"Admin Subject: {result.admin_email.subject}")
        else:
            print("❌ Email operation failed.")
            print(f"Message: {result.message}")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    # Force UTF-8 output on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    asyncio.run(test_send_email())
