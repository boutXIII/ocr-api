# =========================================
# Imports
# =========================================
import os

# =========================================
# API Metadata
# =========================================
PROJECT_NAME: str = "docTR API template"
PROJECT_DESCRIPTION: str = "Template API for Optical Character Recognition"
VERSION: str = "0.0.1"
DEBUG: bool = os.environ.get("DEBUG", "") != "False"
