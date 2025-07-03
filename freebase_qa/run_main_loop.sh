#!/bin/bash

for i in {1..20}  # 无限尝试（你可以换成 while true）
do
    echo "=== Starting round $i ==="
    python -u main.py --LLM gpt-4o-mini --engine azure_openai --method rage

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "main.py finished successfully, exiting loop."
        break
    else
        echo "main.py exited with error code $exit_code. Restarting..."
        sleep 3  # 防止无限重启太快
    fi
done
