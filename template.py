import os

list_of_files = [

    # GitHub Actions
    ".github/workflows/ci.yaml",

    # Notebooks
    "notebooks/EDA.ipynb",

    # Source code
    "src/__init__.py",

    # Logger + Exception (single file)
    "src/logger.py",

    # Utils
    "src/utils/__init__.py",
    "src/utils/common.py",

    # Components
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",

    # Pipelines
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    "config/config.yaml",
    "config/params.yaml",

    "artifacts/.gitkeep",

    "templates/index.html",
    "templates/result.html",
    "static/style.css",

    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    "README.md",
    ".gitignore"
]

for file_path in list_of_files:
    file_path = os.path.join( file_path)
    dir_name = os.path.dirname(file_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
