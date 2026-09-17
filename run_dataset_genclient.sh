#!/usr/bin/env bash

PYTHON_SCRIPT="generate.py"

while true; do
       
    sleep 20

        
    echo "Running Python client..."
    python3 "$PYTHON_SCRIPT"
    PYTHON_EXIT=$?


    echo "Python exited with code $PYTHON_EXIT"

    # Exit if simulation finished successfully
    if [ $PYTHON_EXIT -eq 0 ]; then
        echo "Simulation completed."
        break
    fi


    echo "Restarting in 10 seconds..."
    sleep 10
done
