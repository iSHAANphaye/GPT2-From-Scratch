"""
Web Server for GPT-2 Interactive Interface
Supports running via Flask, with an automatic fallback to Python's standard library ThreadingHTTPServer.
"""
import os
import json
import sys

# Ensure local directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_engine import engine

HOST = "0.0.0.0"
PORT = 5000

try:
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__, static_folder="static", template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status", methods=["GET"])
    def status():
        return jsonify(engine.get_status())

    @app.route("/api/generate", methods=["POST"])
    def generate_api():
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "")
        max_tokens = int(data.get("max_new_tokens", 50))
        temperature = float(data.get("temperature", 0.7))
        top_k = int(data.get("top_k", 50))
        result = engine.generate_text(prompt, max_tokens, temperature, top_k)
        return jsonify(result)

    @app.route("/api/classify", methods=["POST"])
    def classify_api():
        data = request.get_json(force=True) or {}
        text = data.get("text", "")
        result = engine.classify_text(text)
        return jsonify(result)

    @app.route("/api/instruct", methods=["POST"])
    def instruct_api():
        data = request.get_json(force=True) or {}
        instruction = data.get("instruction", "")
        input_context = data.get("input", "")
        max_tokens = int(data.get("max_new_tokens", 120))
        temperature = float(data.get("temperature", 0.0))
        result = engine.instruct_assistant(instruction, input_context, max_tokens, temperature)
        return jsonify(result)

    def run_server():
        print("="*60)
        print(f" * GPT-2 Web Interface running on http://localhost:{PORT}")
        print(f" * Device: {engine.get_status()['device']} ({engine.get_status()['gpu_name']})")
        print("="*60)
        app.run(host=HOST, port=PORT, debug=False)

except ImportError:
    # Standard library fallback server (threading enabled, zero external dependencies)
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse

    class FallbackHandler(BaseHTTPRequestHandler):
        def _send_json(self, data, code=200):
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(response_bytes)
            self.wfile.flush()

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            base_dir = os.path.dirname(os.path.abspath(__file__))

            if parsed.path == "/api/status":
                self._send_json(engine.get_status())
            elif parsed.path in ("/", "/index.html"):
                filepath = os.path.join(base_dir, "templates", "index.html")
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    self.wfile.flush()
                else:
                    self.send_error(404, "Template index.html not found")
            elif parsed.path.startswith("/static/"):
                rel_path = parsed.path.replace("/static/", "")
                filepath = os.path.join(base_dir, "static", rel_path)
                if os.path.exists(filepath):
                    ctype = "text/css" if filepath.endswith(".css") else "application/javascript"
                    with open(filepath, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    self.wfile.flush()
                else:
                    self.send_error(404, "File not found")
            else:
                self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length).decode("utf-8")
                try:
                    data = json.loads(body)
                except Exception:
                    data = {}
            else:
                data = {}

            if parsed.path == "/api/generate":
                prompt = data.get("prompt", "")
                max_tokens = int(data.get("max_new_tokens", 50))
                temperature = float(data.get("temperature", 0.7))
                top_k = int(data.get("top_k", 50))
                result = engine.generate_text(prompt, max_tokens, temperature, top_k)
                self._send_json(result)
            elif parsed.path == "/api/classify":
                text = data.get("text", "")
                result = engine.classify_text(text)
                self._send_json(result)
            elif parsed.path == "/api/instruct":
                instruction = data.get("instruction", "")
                input_context = data.get("input", "")
                max_tokens = int(data.get("max_new_tokens", 120))
                temperature = float(data.get("temperature", 0.0))
                result = engine.instruct_assistant(instruction, input_context, max_tokens, temperature)
                self._send_json(result)
            else:
                self.send_error(404)

    def run_server():
        print("="*60)
        print(" * Running with ThreadingHTTPServer (Built-in Standard Library)")
        print(f" * GPT-2 Web Interface running on http://localhost:{PORT}")
        print(f" * Device: {engine.get_status()['device']} ({engine.get_status()['gpu_name']})")
        print("="*60)
        server = ThreadingHTTPServer((HOST, PORT), FallbackHandler)
        server.serve_forever()


if __name__ == "__main__":
    run_server()
