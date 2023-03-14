#!/bin/sh


echo "running FastAPI server.."

uvicorn src.main:app --host 0.0.0.0 --port 80