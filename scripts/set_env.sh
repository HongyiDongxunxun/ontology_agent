#!/bin/bash
# Source this file to set API keys for the evaluation
export DEEPSEEK_API_KEY_EXTRACTION="${DEEPSEEK_API_KEY_EXTRACTION:-}"
export DEEPSEEK_API_KEY_CLASSIFICATION="${DEEPSEEK_API_KEY_CLASSIFICATION:-}"
export DEEPSEEK_API_KEY_REVIEWER="${DEEPSEEK_API_KEY_REVIEWER:-}"
export LLM_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
echo "API environment variables set."

