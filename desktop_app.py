"""
Desktop launcher for the HADR Translator.

This starts the Flask API in server.py, waits for it to become reachable, and
opens hadr_translator.html in a desktop window via pywebview.
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "hadr_translator.html"
SERVER_FILE = ROOT / "server.py"


def wait_for_server(url: str, timeout: int, process: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before becoming ready with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)

    raise RuntimeError(f"Server did not become ready within {timeout}s: {last_error}")


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    command = [
        sys.executable,
        str(SERVER_FILE),
        "--model_dir",
        str(model_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]

    if args.base_model:
        command.extend(["--base_model", args.base_model])
    if args.whisper:
        command.extend(["--whisper", args.whisper])

    process = subprocess.Popen(command, cwd=str(ROOT))
    atexit.register(stop_server, process)
    return process


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def open_window(url: str) -> bool:
    try:
        import webview
    except ImportError:
        print("pywebview is not installed; opening in the default browser instead.")
        webbrowser.open(url)
        return False

    webview.create_window(
        "HADR Translator",
        url,
        width=1280,
        height=820,
        min_size=(980, 640),
    )
    webview.start()
    return True


def keep_browser_server_alive(url: str, process: subprocess.Popen) -> None:
    print(f"HADR Translator is running at {url}")
    print("Keep this window open while using the app. Press Ctrl+C to stop.")
    try:
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the HADR Translator desktop app")
    parser.add_argument(
        "--model_dir",
        default="./merged_model",
        help="Path to the merged Qwen model or adapter directory",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model name, only needed for LoRA-only adapter directories",
    )
    parser.add_argument(
        "--whisper",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for offline speech-to-text",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--startup_timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HTML_FILE.exists():
        raise FileNotFoundError(f"Missing UI file: {HTML_FILE}")
    if not SERVER_FILE.exists():
        raise FileNotFoundError(f"Missing API server: {SERVER_FILE}")

    server = start_server(args)
    health_url = f"http://127.0.0.1:{args.port}/health"
    app_url = f"http://127.0.0.1:{args.port}/"

    try:
        wait_for_server(health_url, args.startup_timeout, server)
        opened_native_window = open_window(app_url)
        if not opened_native_window:
            keep_browser_server_alive(app_url, server)
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
