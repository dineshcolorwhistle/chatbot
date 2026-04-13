"""One-time script to trigger PDF document ingestion into Pinecone."""

import asyncio
import httpx


async def main():
    print("Starting document ingestion...")
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post("http://localhost:8000/api/admin/ingest")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
