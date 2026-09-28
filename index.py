import os
import mimetypes
from http.server import BaseHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class handler(BaseHTTPRequestHandler):
    """
    Vercel Serverless Function entrypoint.
    Serves the OmniTwin Digital Twin dashboard and assets.
    """
    def do_GET(self):
        path = self.path.split("?")[0].strip("/")
        if not path:
            path = "index.html"

        file_path = os.path.join(BASE_DIR, path)
        if not os.path.isfile(file_path):
            file_path = os.path.join(BASE_DIR, "public", path)

        if not os.path.isfile(file_path):
            file_path = os.path.join(BASE_DIR, "index.html")

        ctype, _ = mimetypes.guess_type(file_path)
        if not ctype:
            if file_path.endswith(".js"):
                ctype = "application/javascript"
            elif file_path.endswith(".json"):
                ctype = "application/json"
            elif file_path.endswith(".css"):
                ctype = "text/css"
            else:
                ctype = "text/html"

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())


def app(environ, start_response):
    """WSGI compatibility handler."""
    path = environ.get("PATH_INFO", "/").strip("/")
    if not path:
        path = "index.html"

    file_path = os.path.join(BASE_DIR, path)
    if not os.path.isfile(file_path):
        file_path = os.path.join(BASE_DIR, "public", path)

    if not os.path.isfile(file_path):
        file_path = os.path.join(BASE_DIR, "index.html")

    ctype, _ = mimetypes.guess_type(file_path)
    if not ctype:
        ctype = "application/javascript" if file_path.endswith(".js") else ("application/json" if file_path.endswith(".json") else "text/html")

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        start_response("200 OK", [("Content-Type", ctype), ("Content-Length", str(len(data)))])
        return [data]
    except Exception as e:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [str(e).encode()]

application = app
