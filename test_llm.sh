#!/bin/bash

# Test LLM with video analysis
# Makes the streaming output more human-readable

echo "Testing NVIDIA Cosmos Reason2-8B model..."
echo "==========================================="
echo ""

curl -s http://0.0.0.0:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/cosmos-reason2-8b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this video?"},
            {"type": "video_url", "video_url": {"url": "https://download.samplelib.com/mp4/sample-5s.mp4"}}
        ]
    }],
    "media_io_kwargs": {
      "video": {
        "num_frames": 10
      }
    },
    "stream": false
  }' | jq -r '
    if .error then
      "ERROR: " + (.error.message // .error)
    elif .choices then
      "Model: " + .model + "\n" +
      "Response:\n" +
      "----------------------------------------\n" +
      .choices[0].message.content + "\n" +
      "----------------------------------------\n" +
      "Finish reason: " + .choices[0].finish_reason + "\n" +
      "Tokens used: " + (.usage.total_tokens // 0 | tostring)
    else
      .
    end
  '

echo ""
echo "Test complete."
