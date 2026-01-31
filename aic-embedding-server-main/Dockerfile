FROM python:3.12-bookworm
WORKDIR /app
COPY . .
RUN pip install --no-cache -r requirements.txt
EXPOSE 50051
CMD [ "python", "main.py" ]
