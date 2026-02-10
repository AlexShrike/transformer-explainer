#!/usr/bin/env python3
"""Entry point — start the Transformer Explainer server."""

import webbrowser
import threading
import uvicorn

def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8070")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("api:app", host="0.0.0.0", port=8070, reload=False)
