import os
import requests
from concurrent import futures
import grpc
import text_embed_pb2
import text_embed_pb2_grpc


class TextEmbeddingService(text_embed_pb2_grpc.TextEmbeddingServiceServicer):
    def GetEmbedding(self, request, context):
        print("embedding request...")
        text: str = request.text
        embeddings = embed_text(text)
        print(len(embeddings))
        return text_embed_pb2.EmbeddingResponse(embeddings=embeddings)


def embed_text(text: str) -> list[float]:
    response = requests.post(os.environ["NGROK_URL"], json={"text": text})
    return response.json()["embeddings"]


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text_embed_pb2_grpc.add_TextEmbeddingServiceServicer_to_server(
        TextEmbeddingService(), server
    )
    port = "[::]:50051"
    server.add_insecure_port(port)
    print(f"### SERVER RUNNING AT {port} ###")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
