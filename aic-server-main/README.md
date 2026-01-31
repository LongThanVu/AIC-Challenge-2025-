---
title: AIC Server
emoji: 😡
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 8080
---

# 🚀 AIC Server

A high-performance backend service built with **Rust**.  
Easily deployable with **Cargo** or **Docker**, with a built-in Swagger UI for API exploration.

---

## 📦 Build & Run

### 🔨 Option 1: Build with Cargo

```bash
cargo build --release
./target/release/aic-server
```

### 🐳 Option 2: Build with Docker

```bash
docker build -t aic-server .
docker run -p 8080:8080 \
  -e DATABASE_URL="your_database_url_here" \
  -e QDRANT_URL="http://localhost:6333" \
  aic-server
```

---

## ⚙️ Environment Variables

Before running, make sure to set:

```bash
export DATABASE_URL="postgres://user:password@localhost:5432/db_name"
export QDRANT_URL="http://localhost:6334"
```

- `DATABASE_URL` → PostgreSQL database connection string
- `QDRANT_URL` → Qdrant vector database endpoint (6333 for HTTP vs. 6334 for gRPC)

---

## 📖 API Documentation

After starting the server, you can explore the API via **Swagger UI**:

👉 [http://localhost:8080/swagger-ui](http://localhost:8080/swagger-ui)

---

## 🗂️ Project Structure

```text
AIC Server/
├── src/              # Source code
├── Cargo.toml        # Rust dependencies
├── Dockerfile        # Container build file
└── README.md         # You're here
```

---

## ✅ Features

- ⚡ Fast & memory-safe (built with Rust)
- 🐳 Dockerized for easy deployment
- 📚 Interactive Swagger UI
- 🗄️ PostgreSQL database support
- 🔍 Qdrant vector database integration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
