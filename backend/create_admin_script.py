import asyncio
import logging
from config import app_config, mongo_config
from services.admin_store import admin_store
from services.auth_service import AuthService

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    try:
        default_email = "dinesh@colorwhistle.com"
        logger.info(f"Checking for existing admin: {default_email}")
        existing = await admin_store.get_by_email(default_email)
        if not existing:
            logger.info("Admin not found. Hashing password...")
            default_pass_hash = AuthService.get_password_hash("Dinesh@#12312")
            logger.info("Creating admin...")
            await admin_store.create_admin(
                name="Dinesh",
                email=default_email,
                password_hash=default_pass_hash
            )
            logger.info("Default admin created: %s", default_email)
        else:
            logger.info("Default admin already exists: %s", default_email)
            
        # Verify
        logger.info("Verifying...")
        admin = await admin_store.get_by_email(default_email)
        logger.info(f"Verified: {admin}")
    except Exception as e:
        logger.exception("Failed to initialize default admin: %s", e)

if __name__ == "__main__":
    asyncio.run(main())
