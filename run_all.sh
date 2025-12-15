#!/bin/bash
#
# Data Drift Analysis - Full Reproducibility Script
# =================================================
# This script runs all analyses and generates all figures and tables
# for the multi-dataset ICU severity score drift study.
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --setup      # Only setup environment
#   ./run_all.sh --analysis   # Only run analysis
#   ./run_all.sh --figures    # Only generate figures
#
# Requirements:
#   - Python 3.9+
#   - uv package manager (install with: pip install uv)
#
# Outputs:
#   - output/all_datasets_drift_results.csv  (full results)
#   - output/all_datasets_drift_deltas.csv   (drift changes)
#   - figures/fig1-7*.png                    (visualizations)
#
# Author: Data Drift Analysis Project
# Date: 2024-12

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DATA DRIFT ANALYSIS - FULL PIPELINE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print step headers
print_step() {
    echo ""
    echo -e "${GREEN}[$1/${TOTAL_STEPS}] $2${NC}"
    echo "----------------------------------------"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate venv
activate_venv() {
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
            source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
        else
            source .venv/bin/activate
        fi
    fi
}

# Parse arguments
RUN_SETUP=true
RUN_ANALYSIS=true
RUN_FIGURES=true
TOTAL_STEPS=6

if [ "$1" == "--setup" ]; then
    RUN_ANALYSIS=false
    RUN_FIGURES=false
    TOTAL_STEPS=3
elif [ "$1" == "--analysis" ]; then
    RUN_SETUP=false
    RUN_FIGURES=false
    TOTAL_STEPS=1
elif [ "$1" == "--figures" ]; then
    RUN_SETUP=false
    RUN_ANALYSIS=false
    TOTAL_STEPS=1
fi

STEP=0

# ============================================================
# STEP 1: Check Prerequisites
# ============================================================
if [ "$RUN_SETUP" = true ]; then
    STEP=$((STEP + 1))
    print_step $STEP "Checking Prerequisites"

    # Check for Python
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}ERROR: Python not found. Please install Python 3.9+${NC}"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    echo "Python version: $PYTHON_VERSION"

    # Check for uv
    if ! command_exists uv; then
        echo -e "${YELLOW}uv not found. Installing uv...${NC}"
        pip install uv
    fi
    echo "uv version: $(uv --version)"

    # ============================================================
    # STEP 2: Setup Virtual Environment
    # ============================================================
    STEP=$((STEP + 1))
    print_step $STEP "Setting Up Virtual Environment"

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv
    else
        echo "Virtual environment already exists"
    fi

    activate_venv
    echo "Virtual environment activated"

    # ============================================================
    # STEP 3: Install Dependencies
    # ============================================================
    STEP=$((STEP + 1))
    print_step $STEP "Installing Dependencies"

    uv pip install -r requirements.txt
    echo "Dependencies installed successfully"
fi

# ============================================================
# STEP 4: Run Batch Analysis
# ============================================================
if [ "$RUN_ANALYSIS" = true ]; then
    STEP=$((STEP + 1))
    print_step $STEP "Running Batch Drift Analysis"

    activate_venv

    echo "Analyzing all datasets..."
    echo ""
    echo "Primary Datasets:"
    echo "  - MIMIC-IV (85,242 patients)"
    echo "  - Amsterdam ICU (27,259 patients)"
    echo "  - Zhejiang ICU (7,932 patients)"
    echo "  - eICU (289,503 patients)"
    echo "  - eICU-New (371,855 patients)"
    echo ""

    python code/batch_analysis.py

    echo ""
    echo "MIMIC-IV Subsets (SOFA + Care Frequency):"
    echo "  - MIMIC-IV Mouthcare (8,675 patients)"
    echo "  - MIMIC-IV Mech. Vent. (8,919 patients)"
    echo ""

    python code/supplementary_analysis.py

    echo ""
    echo -e "${GREEN}Batch analysis complete!${NC}"
fi

# ============================================================
# STEP 5: Generate Figures
# ============================================================
if [ "$RUN_FIGURES" = true ]; then
    STEP=$((STEP + 1))
    print_step $STEP "Generating Figures"

    activate_venv

    echo "Generating cross-dataset comparison figures..."
    echo ""

    python code/generate_all_figures.py

    echo ""
    echo -e "${GREEN}Figure generation complete!${NC}"
fi

# ============================================================
# STEP 6: Summary
# ============================================================
STEP=$((STEP + 1))
print_step $STEP "Summary"

echo -e "${GREEN}All tasks completed successfully!${NC}"
echo ""
echo "Generated outputs:"
echo ""

if [ -d "output" ]; then
    echo "  Data Tables (output/):"
    for f in output/*.csv; do
        [ -f "$f" ] && echo "    - $(basename $f)"
    done
fi

echo ""

if [ -d "figures" ]; then
    echo "  Figures (figures/):"
    for f in figures/fig*.png; do
        [ -f "$f" ] && echo "    - $(basename $f)"
    done
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PIPELINE COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Key outputs:"
echo "  - Full Results: output/all_datasets_drift_results.csv"
echo "  - Drift Deltas: output/all_datasets_drift_deltas.csv"
echo "  - Key Figure: figures/fig7_money_figure.png"
echo ""
echo "To regenerate specific outputs:"
echo "  ./run_all.sh --analysis   # Rerun analysis"
echo "  ./run_all.sh --figures    # Regenerate figures"
echo ""
