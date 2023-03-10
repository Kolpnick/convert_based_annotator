import logging
import numpy as np
import time
from os import getenv

from convert_annotator import ConveRTAnnotator
import sentry_sdk
from flask import Flask, jsonify, request


sentry_sdk.init(getenv("SENTRY_DSN"))

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

annotator = ConveRTAnnotator()
logger.info("Annotator is loaded.")


@app.route("/conv_annot_candidate", methods=["POST"])
def respond_candidate():
    start_time = time.time()
    candidates = request.json['candidates']
    history = request.json['history']
    result = annotator.candidate_selection(history, candidates)
    total_time = time.time() - start_time
    logger.info(f"Annotator candidate selection time: {total_time: .3f}s")
    return jsonify(str(result))


@app.route("/conv_annot_response", methods=["POST"])
def respond_response():
    start_time = time.time()
    response = request.json['response']
    result = annotator.response_encoding(response)
    total_time = time.time() - start_time
    logger.info(f"Annotator response encoding time: {total_time: .3f}s")
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8131)
