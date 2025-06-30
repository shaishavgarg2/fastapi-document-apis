#!/bin/bash

echo "Starting Parameter Match API on port 8000..."
python parameter_match_api.py &

echo "Starting Document Relationship Match API on port 8001..."  
python document_relationship_match.py &

echo "Starting Parameter Value Extraction API on port 8002..."
python parameter_value_extraction_api.py &

wait