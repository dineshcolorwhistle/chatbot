"""Script to trigger PDF document ingestion into Pinecone."""

import asyncio
import httpx
import argparse


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument("--namespace", type=str, help="Pinecone namespace to ingest into", default=None)
    parser.add_argument("--dir", type=str, help="Custom documents directory path", default=None)
    args = parser.parse_args()

    payload = {}
    if args.namespace:
        payload["namespace"] = args.namespace
    if args.dir:
        payload["documents_dir"] = args.dir

    print(f"Starting document ingestion... Payload: {payload}")
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post("http://localhost:8000/api/admin/ingest", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
