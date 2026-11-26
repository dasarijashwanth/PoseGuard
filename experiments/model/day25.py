"""Day 25: capture-session launcher.

Day 24 wired the re-benchmark into one command, but the capture
itself still took three manual steps: serve the Day 14 tool on
localhost, follow the checklist at the desk, then find the two
downloaded files and point Day 24 at them. Day 25 is the one
command that does all of that:

1. Serve experiments/rerun-capture.html on localhost (the camera
   needs a secure context, and localhost counts as one; a hosted
   page never gets camera access) and print the local URL.
2. Walk experiments/CAPTURE_CHECKLIST.md in the terminal, in
   order, with the session timeline (calibration, holds and
   timings) taken from experiments.eval.rerun_protocol, so the
   numbers printed here are the numbers the benchmark uses.
3. Watch a drop directory (the browser's Downloads folder by
   default) for the capture CSV and its manifest. When both are
   present, run the Day 24 pipeline on them by calling
   day24.run_day24 as a function, the same pattern Day 24 uses
   for run_day16, and report the archive location and verdict.
4. If the files are not there yet, say exactly what is missing
   and print the manual Day 24 command as the fallback, so the
   session can always be finished by hand.

Usage (from the repo root):
    python -m experiments.model.day25 [--port 8000]
        [--drop-dir <folder>] [--timeout <seconds>]
        [--source <label for the archive>]
        [--out-dir <archive root>]
    python -m experiments.model.day25 --no-wait [--drop-dir <folder>]

--no-wait skips the server and the wait: it checks the drop
directory once and either runs the pipeline (both files there)
or prints what is missing plus the fallback command.

Exit code 0 when the pipeline ran, 2 when it could not (files
missing, port taken, or the capture failed Day 24 validation),
130 when stopped with Ctrl+C. Nothing is benchmarked or archived
by this module itself; Day 24 owns that.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.eval.rerun_protocol import (
    CALIB_S,
    HOLD_S,
    LOSS_S,
    READY_S,
    REST_S,
    TARGET_HEAD_DEG,
    TARGET_SHOULDER_DEG,
    TRIALS_PER_CUE,
    WORK_S,
)

from .day24 import DEFAULT_SOURCE, best_variant, discover_inputs, run_day24

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
CAPTURE_HTML = EXPERIMENTS_DIR / "rerun-capture.html"
CHECKLIST_MD = EXPERIMENTS_DIR / "CAPTURE_CHECKLIST.md"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_S = 1800.0
POLL_S = 2.0


# ----------------------------------------------------------------
# Serving the capture page
# ----------------------------------------------------------------

def make_handler(html_path: Path) -> type[BaseHTTPRequestHandler]:
    """An HTTP handler that serves one file: the capture page.

    The page answers at "/" and at "/rerun-capture.html"; every
    other path is a 404. The file is read per request, so an
    edited capture tool is picked up without a restart.
    """
    page = Path(html_path)

    class CaptureHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib name)
            if self.path.split("?")[0] in ("/", "/rerun-capture.html"):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args) -> None:
            pass  # keep the terminal for the checklist, not HTTP logs

    return CaptureHandler


def start_server(html_path: Path = CAPTURE_HTML,
                 port: int = DEFAULT_PORT):
    """Serve the capture page on 127.0.0.1 in a background thread.

    Returns (server, thread, url). Raises OSError when the port
    is taken; the caller turns that into a clear message.
    """
    server = ThreadingHTTPServer(("127.0.0.1", port),
                                 make_handler(html_path))
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    bound_port = server.server_address[1]
    return server, thread, f"http://localhost:{bound_port}/rerun-capture.html"


# ----------------------------------------------------------------
# The checklist walk
# ----------------------------------------------------------------

def checklist_steps(path: Path = CHECKLIST_MD) -> list[tuple[str, str]]:
    """Parse CAPTURE_CHECKLIST.md into ordered (section, step) pairs.

    Sections come from "## " headings; steps are the numbered
    list items under them, with wrapped lines joined and markdown
    bold markers removed. The order is the file's order: before,
    during, after.
    """
    steps: list[tuple[str, str]] = []
    section = ""
    current: list[str] | None = None

    def flush() -> None:
        if current is not None:
            text = " ".join(part.strip() for part in current)
            steps.append((section, text.replace("**", "")))

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            current = None
            section = line[3:].strip()
        elif re.match(r"^\d+\.\s", line):
            flush()
            current = [re.sub(r"^\d+\.\s*", "", line)]
        elif current is not None and line.strip():
            current.append(line)
        elif not line.strip():
            flush()
            current = None
    flush()
    return steps


def session_timeline() -> list[tuple[str, float]]:
    """The guided session as (label, seconds) blocks, in order.

    Every number comes from experiments.eval.rerun_protocol, the
    same constants the benchmark scores against, so the walk
    cannot drift from the protocol.
    """
    blocks: list[tuple[str, float]] = [
        (f"Calibration: sit normally, upright, looking at the "
         f"screen", CALIB_S),
    ]
    for cue, target in (("head", TARGET_HEAD_DEG),
                        ("shoulder", TARGET_SHOULDER_DEG)):
        per_hold = READY_S + HOLD_S + REST_S
        blocks.append((
            f"{cue.capitalize()} holds x{TRIALS_PER_CUE}: "
            f"{READY_S:g} s ready, {HOLD_S:g} s hold past the "
            f"{target:g} deg target, {REST_S:g} s rest, each hold",
            per_hold * TRIALS_PER_CUE))
    blocks.append(("Work sample: work normally at the desk, "
                   "ordinary drift is the point", WORK_S))
    blocks.append(("Landmark-loss test: follow the on-screen "
                   "prompt", LOSS_S))
    return blocks


def format_walk(steps: list[tuple[str, str]],
                timeline: list[tuple[str, float]]) -> str:
    """The terminal walk: checklist steps, then the timeline."""
    lines = ["Capture checklist (experiments/CAPTURE_CHECKLIST.md):"]
    last_section = None
    number = 0
    for section, text in steps:
        if section != last_section:
            lines.append(f"\n{section}:")
            last_section = section
            number = 0
        number += 1
        lines.append(f"  {number}. {text}")
    lines.append("\nSession timeline (the tool prompts you through "
                 "each block):")
    total = 0.0
    for label, seconds in timeline:
        total += seconds
        lines.append(f"  - {label} [{seconds:g} s]")
    lines.append(f"  Total guided time: about {total / 60:.1f} min "
                 f"({total:g} s).")
    return "\n".join(lines)


# ----------------------------------------------------------------
# Watching the drop directory
# ----------------------------------------------------------------

def scan_drop_dir(drop_dir: str) -> dict:
    """What the drop directory holds right now.

    Returns {"state": ..., "csv": path or None, "manifest": path
    or None}. States: "empty" (neither file), "csv_only",
    "manifest_only", "ready" (both). Pairing reuses Day 24's
    discover_inputs: newest .csv, manifest by same stem else
    newest .json.
    """
    folder = Path(drop_dir)
    if not folder.is_dir():
        return {"state": "empty", "csv": None, "manifest": None}
    has_csv = any(folder.glob("*.csv"))
    has_json = any(folder.glob("*.json"))
    if has_csv and has_json:
        csv_path, manifest_path = discover_inputs(str(folder))
        return {"state": "ready", "csv": csv_path,
                "manifest": manifest_path}
    if has_csv:
        csv_path, _ = discover_inputs(str(folder))
        return {"state": "csv_only", "csv": csv_path,
                "manifest": None}
    if has_json:
        return {"state": "manifest_only", "csv": None,
                "manifest": None}
    return {"state": "empty", "csv": None, "manifest": None}


def fallback_command(drop_dir: str) -> str:
    """The exact manual Day 24 command for finishing by hand."""
    return (f"python -m experiments.model.day24 "
            f"--drop-dir {drop_dir}")


def _missing_line(scan: dict) -> str:
    if scan["state"] == "csv_only":
        return (f"Capture CSV found ({scan['csv']}); still "
                f"waiting for the manifest .json. In the capture "
                f"tool, press Download manifest too.")
    if scan["state"] == "manifest_only":
        return ("A .json is in the folder but no capture .csv "
                "yet. In the capture tool, press Download CSV.")
    return ("No capture files yet. Finish the session in the "
            "browser, then press Download CSV and Download "
            "manifest.")


def wait_for_capture(drop_dir: str,
                     timeout_s: float = DEFAULT_TIMEOUT_S,
                     poll_s: float = POLL_S,
                     sleep=time.sleep,
                     now=time.monotonic) -> tuple[str, str] | None:
    """Poll the drop directory until both files are present.

    Prints each state change once. Returns (csv, manifest) when
    ready, None on timeout. timeout_s <= 0 waits until Ctrl+C.
    sleep and now are injectable so tests never really wait.
    """
    deadline = None if timeout_s <= 0 else now() + timeout_s
    last_state = None
    while True:
        scan = scan_drop_dir(drop_dir)
        if scan["state"] == "ready":
            return scan["csv"], scan["manifest"]
        if scan["state"] != last_state:
            print(_missing_line(scan))
            last_state = scan["state"]
        if deadline is not None and now() >= deadline:
            return None
        sleep(poll_s)


# ----------------------------------------------------------------
# The Day 24 handoff
# ----------------------------------------------------------------

def run_pipeline(csv_path: str, manifest_path: str | None,
                 source: str = DEFAULT_SOURCE,
                 out_dir: str | None = None) -> dict:
    """Run Day 24 on the capture and report where the verdict went.

    run_day24 prints the pre-flight and the full summary itself;
    this adds the short closing lines: archive location, the
    Day 16 verdict, and the best combined variant per cue.
    Raises ValueError when the capture fails validation (Day 24
    archives nothing then).
    """
    archive = run_day24(csv_path, manifest_path, source=source,
                        out_dir=out_dir)
    print(f"\nCapture session complete. Archive: "
          f"{archive['run_dir']}")
    print(f"Day 16 verdict: {archive['day16']['verdict']}")
    for cue in ("head", "shoulder"):
        name, best = best_variant(archive["combined"]["cues"][cue])
        print(f"Combined {cue}: {name} at threshold "
              f"{best['threshold']}, {best['detected']}/"
              f"{best['trials_total']} trials, "
              f"{best['false_positives']} false positives, "
              f"F1 {best['f1']:.3f}: "
              f"{'passes the gate' if best['pass'] else 'does not pass the gate'}.")
    return archive


def _not_ready_message(drop_dir: str, scan: dict) -> str:
    return (f"{_missing_line(scan)}\n"
            f"Drop folder: {drop_dir}\n"
            f"When both files are there, finish by hand with:\n"
            f"  {fallback_command(drop_dir)}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 25: capture-session launcher")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="localhost port for the capture page "
                             "(default: 8000, the checklist port)")
    parser.add_argument("--drop-dir", default=None,
                        help="folder the browser downloads land in "
                             "(default: ~/Downloads)")
    parser.add_argument("--timeout", type=float,
                        default=DEFAULT_TIMEOUT_S,
                        help="seconds to wait for the two files "
                             "(default: 1800; 0 waits until Ctrl+C)")
    parser.add_argument("--no-wait", action="store_true",
                        help="check the drop folder once, no server, "
                             "no waiting")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="label for the data in the archive "
                             "(default: Run 2 re-run capture)")
    parser.add_argument("--out-dir", default=None,
                        help="archive root (default: runs/ next to "
                             "the Day 24 module)")
    args = parser.parse_args(argv)

    drop_dir = args.drop_dir or str(Path.home() / "Downloads")
    Path(drop_dir).mkdir(parents=True, exist_ok=True)

    try:
        scan = scan_drop_dir(drop_dir)
        if scan["state"] == "ready":
            print(f"Capture files already in {drop_dir}; "
                  f"running the Day 24 pipeline.")
            run_pipeline(scan["csv"], scan["manifest"],
                         source=args.source, out_dir=args.out_dir)
            return 0
        if args.no_wait:
            print(_not_ready_message(drop_dir, scan))
            return 2

        try:
            server, _thread, url = start_server(port=args.port)
        except OSError as e:
            print(f"Day 25 could not serve the capture page on "
                  f"port {args.port}: {e}. Pick another port "
                  f"with --port, or finish by hand:\n"
                  f"  {fallback_command(drop_dir)}",
                  file=sys.stderr)
            return 2
        try:
            print(f"Capture page: {url}")
            print("Open it in Chrome or Edge and allow camera "
                  "access. (The launcher replaces the checklist's "
                  "serve-it-yourself step.)\n")
            print(format_walk(checklist_steps(), session_timeline()))
            print(f"\nWatching {drop_dir} for the capture CSV "
                  f"and manifest...")
            found = wait_for_capture(drop_dir,
                                     timeout_s=args.timeout)
        finally:
            server.shutdown()
            server.server_close()
        if found is None:
            print(_not_ready_message(drop_dir,
                                     scan_drop_dir(drop_dir)))
            return 2
        csv_path, manifest_path = found
        run_pipeline(csv_path, manifest_path, source=args.source,
                     out_dir=args.out_dir)
        return 0
    except KeyboardInterrupt:
        print(f"\nStopped. When both files are in {drop_dir}, "
              f"finish by hand with:\n"
              f"  {fallback_command(drop_dir)}")
        return 130
    except (ValueError, OSError) as e:
        # Same convention as Day 24: a capture that fails
        # validation (or a file that vanished) is a user error
        # with a clear message, exit code 2, no traceback.
        print(f"Day 25 failed:\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
