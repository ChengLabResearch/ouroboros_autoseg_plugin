"""Bounded volume-service fixture for the standalone GPU smoke, never production.

The backend uses the production HTTP transfer protocol. Only this run's input
and outputs are accepted; the logical volume maps to the harness's Docker volume.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess


class SmokeTransfers:
    def __init__(self, input_path, output_paths, volume_name, image, events_path):
        self.input_path = Path(input_path)
        self.output_paths = [Path(path) for path in output_paths]
        self.volume_name = volume_name
        self.image = image
        self.events_path = Path(events_path)

    def transfer(self, operation, request):
        if request.get("volumeName") != "ouroboros-volume":
            raise ValueError("Unexpected logical volume")
        if request.get("pluginFolderName") != "sam3-segmentation":
            raise ValueError("Unexpected plugin folder")
        if operation == "copy-to-volume":
            allowed = {str(self.input_path): ""}
        elif operation == "copy-to-host":
            allowed = {str(path): "Segmentation" for path in self.output_paths}
        else:
            raise ValueError("Unexpected transfer operation")

        files = request.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Expected nonempty files list")
        for mapping in files:
            if not isinstance(mapping, dict) or mapping.get("sourcePath") not in allowed:
                raise ValueError("Unexpected host path")
            if mapping.get("targetPath") != allowed[mapping["sourcePath"]]:
                raise ValueError("Unexpected volume folder")

        for mapping in files:
            self.copy(operation, Path(mapping["sourcePath"]), mapping["targetPath"])
        with self.events_path.open("a", encoding="utf-8") as events:
            events.write(json.dumps({"operation": operation, "files": files}) + "\n")

    def copy(self, operation, host_path, target_folder):
        to_volume = operation == "copy-to-volume"
        volume_folder = f"/volume/sam3-segmentation/{target_folder}"
        copy_command = (
            'mkdir -p "$VOLUME_FOLDER" && cp -r "/host/$FILE_NAME" "$VOLUME_FOLDER/$FILE_NAME"'
            if to_volume else
            'cp -r "$VOLUME_FOLDER/$FILE_NAME" "/host/$FILE_NAME"'
        )
        subprocess.run(
            [
                "docker", "run", "--rm", "--entrypoint", "/bin/sh",
                "-v", f"{host_path.parent}:/host" + (":ro" if to_volume else ""),
                "-v", f"{self.volume_name}:/volume",
                "-e", f"FILE_NAME={host_path.name}",
                "-e", f"VOLUME_FOLDER={volume_folder}",
                self.image, "-ceu", copy_command,
            ],
            check=True, capture_output=True, text=True,
        )


def make_server(transfers, address=("0.0.0.0", 0)):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, message):
            body = message.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == "/" else 404, "Smoke volume fixture")

        def do_POST(self):
            operation = self.path.removeprefix("/")
            if operation not in {"copy-to-volume", "copy-to-host"}:
                self.reply(404, "Unknown operation")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected object request")
                transfers.transfer(operation, request)
            except (ValueError, TypeError) as error:
                self.reply(400, str(error))
            except subprocess.CalledProcessError as error:
                self.reply(500, f"Docker transfer failed: {error.stderr}")
            except OSError as error:
                self.reply(500, f"Transfer failed: {error}")
            else:
                self.reply(200, "Files copied successfully.")

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--events-file", required=True)
    args = parser.parse_args()
    transfers = SmokeTransfers(args.input, args.output, args.volume, args.image, args.events_file)
    with make_server(transfers) as server:
        Path(args.ready_file).write_text(json.dumps({"port": server.server_port}), encoding="utf-8")
        server.serve_forever()


if __name__ == "__main__":
    main()
