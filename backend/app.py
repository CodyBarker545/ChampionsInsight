"""Starts and configures the Champions Insight Flask backend."""

import logging
import threading
import os

from flask import Flask
from flask_cors import CORS

from api.routes import api_bp
from pokemon_damage_calculator.backend_process import start_damage_calc_backend


logger = logging.getLogger(__name__)


def warm_computer_vision_services():
    """Warm the actual upload models before the first photo arrives."""
    try:
        import numpy as np

        from services import cv_service
        from services.pokemon_mobilenetv3_classifier_service import predict_pokemon_crops

        model = cv_service.load_yolo_slot_object_model()
        if model is not None:
            model.predict(source=np.zeros((220, 640, 3), dtype=np.uint8),
                          conf=cv_service.YOLO_SLOT_OBJECT_CONFIDENCE,
                          imgsz=cv_service.YOLO_SLOT_OBJECT_IMAGE_SIZE,
                          device=cv_service.get_yolo_inference_device(), verbose=False)
        predict_pokemon_crops([np.zeros((224, 224, 3), dtype=np.uint8)])
        logger.info("Computer vision services warmed.")
    except Exception as error:
        logger.warning("Computer vision warmup failed: %s", error)


def warm_rag_services():
    """Load RAG models and index so the first mechanics question is faster."""
    try:
        from rag.embedding_model import load_embedding_model
        from rag.search_faiss import CHUNKS_PATH, INDEX_PATH, load_chunk_records, load_search_index

        load_embedding_model()
        if INDEX_PATH.exists():
            load_search_index(INDEX_PATH)
        if CHUNKS_PATH.exists():
            load_chunk_records(CHUNKS_PATH)
        logger.info("RAG services warmed.")
    except Exception as error:
        logger.warning("RAG warmup failed: %s", error)


def warm_battle_advice_services():
    """Load tournament advice JSON summaries so first advice requests are faster."""
    try:
        from services.battle_advice_cache_service import warm_battle_advice_cache

        warm_battle_advice_cache()
        logger.info("Battle advice caches warmed.")
    except Exception as error:
        logger.warning("Battle advice warmup failed: %s", error)


def start_background_warmup():
    def warm_capture_then_rag():
        # Prioritize capture models before competing transformer initialization.
        warm_computer_vision_services()
        warm_rag_services()

    for target, thread_name in (
        (warm_capture_then_rag, "model-warmup"),
        (warm_battle_advice_services, "battle-advice-warmup"),
    ):
        warmup_thread = threading.Thread(
            target=target,
            name=thread_name,
            daemon=True,
        )
        warmup_thread.start()


# Creates and configures the Flask application.
def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(api_bp, url_prefix="/api")
    if os.environ.get("CHAMPIONS_SKIP_WARMUP") != "1":
        start_background_warmup()
    return app


if __name__ == "__main__":
    # Establish the local calculator before CV/RAG warmup competes for resources.
    start_damage_calc_backend()

app = create_app()


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        ssl_context="adhoc",
        use_reloader=False,
    )
