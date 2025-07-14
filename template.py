import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define project root
project_root = "automated_procurement"

# Define core directories from the project structure
directories = [
    "artifacts/data/encode",
    "artifacts/evaluation",
    "artifacts/models",
    "artifacts/predictions",
    "artifacts/samples",
    
    "data/processed",
    "data/raw",
    
    "mlruns",
    
    "pipelines",
    
    "src",
    
    "steps",
    
    "utils"
]

# Create each directory
for dir in directories:
    dir_path = Path(project_root) / dir
    dir_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {dir_path}")
