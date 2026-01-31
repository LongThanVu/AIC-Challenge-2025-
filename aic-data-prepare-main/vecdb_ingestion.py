import asyncio
import csv
import os
from pathlib import Path
import re
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import select
from sqlalchemy.orm import Session
from sql_ingestion import Keyframe, Video, get_engine
import numpy as np

client = AsyncQdrantClient(
    host=os.environ["QDRANT_HOST"],
    port=int(os.environ["QDRANT_PORT"]),
    grpc_port=int(os.environ["QDRANT_GRPC_PORT"]),
)


async def ingest_keyframes():
    engine = get_engine()
    with Session(engine) as session:
        points = []
        query = select(
            Keyframe.id, Keyframe.video_related_frame_id, Video.l, Video.v
        ).join(Keyframe.video)
        for (
            keyframe_id,
            keyframe_video_related_frame_id,
            video_l,
            video_v,
        ) in session.execute(query):
            image_embeddings = np.load(
                Path(
                    "prepared-data",
                    "embeddings",
                    "images",
                    f"L{video_l}_V{video_v:03}",
                    f"{keyframe_video_related_frame_id:03}.npy",
                ).open("rb")
            ).squeeze(0)
            object_embeddings = np.load(
                Path(
                    "prepared-data",
                    "embeddings",
                    "objects",
                    f"L{video_l}_V{video_v:03}",
                    f"{keyframe_video_related_frame_id:03}.npy",
                )
            )

            points.append(
                PointStruct(
                    id=keyframe_id,
                    vector={"images": image_embeddings, "objects": object_embeddings},
                )
            )
            if len(points) == 1024:
                print("ingesting batch...")
                await client.upsert("keyframes", points=points)
                points = []

        print("ingesting final batch")
        await client.upsert("keyframes", points=points)


async def main():
    if await client.collection_exists("keyframes"):
        await client.delete_collection("keyframes")
    await client.create_collection(
        "keyframes",
        vectors_config={
            "images": VectorParams(size=640, distance=Distance.COSINE),
            "objects": VectorParams(size=640, distance=Distance.EUCLID),
        },
    )
    await ingest_keyframes()


if __name__ == "__main__":
    asyncio.run(main())
