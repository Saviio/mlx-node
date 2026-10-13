"""Optional default-retry client for audio-sdk-retry.test.ts; no model downloads."""
import json
import sys

import openai

# Keep the installed SDK's default retries and HTTP transport unchanged.
output_format = sys.argv[2] if len(sys.argv) > 2 else "pcm"
with openai.OpenAI(base_url=sys.argv[1], api_key="local-test", timeout=10) as client:
    try:
        with client.audio.speech.with_streaming_response.create(
            model="tts", voice="v", input="Hello.",
            response_format="pcm" if output_format == "sse" else output_format,
            stream_format="sse" if output_format == "sse" else "audio",
            extra_headers={"x-sdk-retry-probe": "true"},
        ) as response:
            result = {"status": response.status_code, "bytes": sum(len(chunk) for chunk in response.iter_bytes())}
    except openai.APIStatusError as error:
        result = {"status": error.status_code, "code": error.code}
    print(json.dumps(result))
