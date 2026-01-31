import json
import re
import csv
from pathlib import Path
from dataclasses import asdict, dataclass
import os
from typing import Iterator, List

from sqlalchemy import ForeignKey, String, UniqueConstraint, create_engine, insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    l: Mapped[int] = mapped_column()
    v: Mapped[int] = mapped_column()
    watch_url: Mapped[str] = mapped_column(String)

    __table_args__ = (UniqueConstraint("l", "v", name="uq_videos_l_v"),)

    keyframes: Mapped[List["Keyframe"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Video(id={self.id!r}, l={self.l!r}, v={self.v!r}, watch_url={self.watch_url!r})"


class Keyframe(Base):
    __tablename__ = "keyframes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    video: Mapped[Video] = relationship()
    video_related_frame_id: Mapped[int] = mapped_column()
    video_related_frame_timestamp: Mapped[float] = mapped_column()

    __table_args__ = (
        UniqueConstraint(
            "video_id", "video_related_frame_id", name="uq_keyframes_video_frame"
        ),
    )

    video: Mapped["Video"] = relationship(back_populates="keyframes")

    def __repr__(self) -> str:
        return (
            f"Keyframe(id={self.id!r}, video_id={self.video_id!r}, "
            f"frame_id={self.video_related_frame_id!r}, ts={self.video_related_frame_timestamp!r})"
        )


@dataclass()
class NewVideo:
    l: int
    v: int
    watch_url: str


@dataclass()
class NewKeyframe:
    video_id: int
    video_related_frame_id: int
    video_related_frame_timestamp: float


def load_videos() -> Iterator[NewVideo]:
    for map_file in Path("data", "media-info").iterdir():
        map_file_name = map_file.name
        pattern = r"\bL(?P<l>\d+)_V(?P<v>\d{3})\.json\b"
        parsed_video_name = re.match(pattern, map_file_name)
        assert parsed_video_name is not None
        l = int(parsed_video_name.group("l"))
        v = int(parsed_video_name.group("v"))
        with map_file.open("r") as f:
            map_file_data = json.load(f)
            watch_url = map_file_data["watch_url"]
        yield NewVideo(l=l, v=v, watch_url=watch_url)


def load_map_keyframes(video_id: int, video_name: str) -> Iterator[NewKeyframe]:
    csv_map_file = Path("data", "map-keyframes", f"{video_name}.csv")
    with csv_map_file.open("r") as f:
        csv_reader = csv.reader(f)
        _ = next(csv_reader)
        for row in csv_reader:
            n = int(row[0])
            pts_time = float(row[1])
            new_keyframe = NewKeyframe(
                video_id=video_id,
                video_related_frame_id=n,
                video_related_frame_timestamp=pts_time,
            )
            yield new_keyframe


def get_engine():
    engine = create_engine(os.environ["DATABASE_URL"], echo=True)
    return engine


def main():
    engine = get_engine()
    with Session(engine) as session:
        for video in load_videos():
            try:
                print(f"ingesting L{video.l}_V{video.v:03}")
                video_id = session.scalars(
                    insert(Video).returning(Video.id, sort_by_parameter_order=True),
                    [asdict(video)],
                ).all()[0]
                new_keyframes = []
                for new_keyframe in load_map_keyframes(
                    video_id, f"L{video.l}_V{video.v:03}"
                ):
                    print(
                        f"ingesting L{video.l}_V{video.v:03}/{new_keyframe.video_related_frame_id:03}.jpg"
                    )
                    new_keyframes.append(asdict(new_keyframe))
                session.scalars(
                    insert(Keyframe).returning(Keyframe, sort_by_parameter_order=True),
                    new_keyframes,
                )
            except Exception as e:
                print(f"error: {e}")
            session.commit()


if __name__ == "__main__":
    main()
