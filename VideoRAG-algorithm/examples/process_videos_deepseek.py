import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag._llm import ollama_config
from videorag import VideoRAG, QueryParam


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    video_paths = [
        "./examples/test1.mp4",
    ]

    videorag = VideoRAG(
        llm=ollama_config,
        working_dir="./videorag-workdir-5090-ollama-videorag",
    )

    # 1. 비디오 인덱싱
    videorag.insert_video(video_path_list=video_paths)

    # 2. caption / visual branch 모델 로드
    videorag.load_caption_model()

    # 3. full VideoRAG query 실행
    response = videorag.query(
        "What do you want to explain in the video?",
        param=QueryParam(mode="videorag"),
    )

    print("\n========== VideoRAG Response ==========\n")
    print(response)
