#!/bin/bash

# Start server
python server/server.py &
SERVER_PID=$!

sleep 3

# Start clients
python client/client.py &
python client/client.py &

wait $SERVER_PID
