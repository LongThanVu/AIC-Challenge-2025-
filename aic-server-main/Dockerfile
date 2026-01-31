FROM rust:1.89-bookworm AS builder-cache
WORKDIR /app/

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/ && echo "fn main() {}" > src/main.rs
RUN cargo build --release

FROM rust:1.89-bookworm AS builder
WORKDIR /app/

RUN apt-get update && apt-get -y install protobuf-compiler
COPY Cargo.toml Cargo.lock build.rs /app/
COPY proto/ /app/proto/
COPY src/ /app/src/
COPY --from=builder-cache /app/target/ /app/target/
RUN cargo build --release

FROM debian:bookworm
WORKDIR /app/

RUN apt-get update && apt-get -y install libpq5
COPY --from=builder /app/target/release/aic-server /app/server

EXPOSE 8080

CMD [ "/app/server" ]