import asyncio
import sys
import os

# Ensure backend root is in path
sys.path.insert(0, os.path.dirname(__file__) + '/../backend')

from models.schemas import Session, ConversationStage
from providers.factory import create_llm_provider
from services.email_agent import EmailAgent

async def test_send_email():
    print("Generating Mock Summary Email...")
    provider = create_llm_provider()
    
    agent = EmailAgent(provider)
    
    session = Session(session_id="test-mock-email-123", stage=ConversationStage.EMAIL)
    session.collected_data.personal_info.name = "Karthick"
    session.collected_data.personal_info.email = "dinesh02121990@gmail.com"
    session.collected_data.tech_discovery.project_type = "WordPress 5 paged website"
    session.summary = "Karthick wants to build a WordPress websites with 5 pages: Home, About Us, Services, Blog, and Contact."
    
    try:
        user_email = await agent._compose_user_email(session)
        admin_email = await agent._compose_admin_email(session)
        
        print("\n=== USER EMAIL ===")
        print(f"Subject: {user_email.subject}")
        print(f"To: {user_email.to}")
        print("-" * 20)
        print(user_email.body)
        
        print("\n=== ADMIN EMAIL ===")
        print(f"Subject: {admin_email.subject}")
        print(f"To: {admin_email.to}")
        print("-" * 20)
        print(admin_email.body)
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    asyncio.run(test_send_email())
