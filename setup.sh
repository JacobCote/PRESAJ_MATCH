#!/bin/bash


# Create the 'data' directory if it doesn't exist
if [ ! -d "data" ]; then
  mkdir data
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo 'venv not existing, crating a venv'
  # Create the virtual environment
  python3 -m venv venv
  # Activate the virtual environment
  source venv/Scripts/activate
  # Install dependencies from requirements.txt
  pip install -r requirements.txt
  # Deactivate the virtual environment
  deactivate
fi

# Activate the virtual environment
#source venv/Scripts/activate
# Run the Python script 'match.py'
#python3 match.py

