"""
Credit Scoring Model Package
Pro National Bank - Data Science Team
"""

__version__ = "1.0.0"
__author__ = "Pro National Bank - Data Science Team"
__email__ = "datascience@pronationalbank.com"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"
