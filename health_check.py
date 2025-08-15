# health_check.py
import sys, json, httpx

URL = "http://127.0.0.1:1234/v1/chat/completions"

def main():
    payload = {
        "model":"gpt-oss-20b",
        "messages":[{"role":"user","content":"Ping"}],
        "max_tokens":5
    }
    try:
        r = httpx.post(URL, json=payload, timeout=5)
        r.raise_for_status()
        print("✅ LLM Studio OK:", r.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        sys.exit(f"❌ Health check failed: {exc}")

if __name__ == "__main__":
    main()
