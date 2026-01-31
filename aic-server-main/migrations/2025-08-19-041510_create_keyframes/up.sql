CREATE TABLE keyframes (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    frame_index BIGINT NOT NULL,
    frame_timestamp REAL NOT NULL,
    video_id BIGINT NOT NULL REFERENCES videos (id),
    UNIQUE (video_id, frame_index)
)
