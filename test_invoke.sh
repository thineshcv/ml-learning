#!/usr/bin/env bash
# Send a simple test request to the running MLflow model server.
# Uses MLflow 2.0 scoring format: one of {inputs,dataframe_records,dataframe_split,instances}.
# Adjust URL/port or feature values as needed.
set -euo pipefail
URL=${1:-http://127.0.0.1:1234/invocations}

echo "Sending test invocation to $URL"
# Use 'inputs' (list of lists) which matches the model's expected 5-feature input
PAYLOAD='{"inputs": [[71382.0, 5.682861322, 7.009188143, 4.09, 23086.8005]]}'

echo "Payload: $PAYLOAD"

# Perform request and capture body + status
RESP=$(curl -s -w "\nHTTPSTATUS:%{http_code}" -X POST -H "Content-Type:application/json" -d "$PAYLOAD" "$URL" || true)
BODY=$(echo "$RESP" | sed -n '1,/HTTPSTATUS:/p' | sed '$d')
STATUS=$(echo "$RESP" | awk -F'HTTPSTATUS:' '{print $2}')

if command -v jq >/dev/null 2>&1; then
	echo "Response body:"
	echo "$BODY" | jq || echo "$BODY"
else
	echo "Response body:"
	echo "$BODY"
fi

echo "HTTP status: $STATUS"

exit 0
