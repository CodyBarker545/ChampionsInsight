"""Starts and configures the Champions Insight Flask backend."""

import logging
import threading

from flask import Flask
from flask_cors import CORS

from api.routes import api_bp


logger = logging.getLogger(__name__)


def warm_computer_vision_services():
    """Loads CV references and the embedding model before the first upload."""
    try:
        import numpy as np

        from services.cv_detection_service import get_detection_service
        from services.embedding_model_service import embed_image

        get_detection_service()
        embed_image(np.zeros((64, 64, 3), dtype=np.uint8))
        logger.info("Computer vision services warmed.")
    except Exception as error:
        logger.warning("Computer vision warmup failed: %s", error)


def start_background_warmup():
    warmup_thread = threading.Thread(
        target=warm_computer_vision_services,
        name="cv-warmup",
        daemon=True,
    )
    warmup_thread.start()


# Creates and configures the Flask application.
def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(api_bp, url_prefix="/api")
    start_background_warmup()
    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        ssl_context="adhoc"
    )
