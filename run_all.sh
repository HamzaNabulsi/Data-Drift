#!/bin/bash
#
# Data Drift Analysis - Full Reproducibility Script
# =================================================
# This script runs all analyses and generates all figures and tables
# for the multi-dataset ICU severity score drift study.
#
# Usage:
#   ./run_all.sh                  # Run everything (default: 100 bootstrap iterations)
#   ./run_all.sh --fast           # Fast mode (~1 min, 2 bootstrap iterations)
#   ./run_all.sh --bootstrap 1000 # Production mode (~2-4 hours, 1000 iterations)
#   ./run_all.sh --setup          # Only setup environment
#   ./run_all.sh --analysis       # Only run analysis
#   ./run_all.sh --figures        # Only generate figures
#   ./run_all.sh --review         # Only generate the review response document
#   ./run_all.sh --sofa-thresholds "2 8 10"  # Custom SOFA thresholds
#
# Bootstrap iterations control confidence interval accuracy:
#   --fast           :   2 iterations (~1 min)   - for testing
#   default          : 100 iterations (~15 min)  - for development
#   --bootstrap 1000 : 1000 iterations (~2-4 hr) - for production/publication
#
# Requirements:
#   - Python 3.10+
#   - uv package manager (install with: pip install uv)
#
# Outputs:
#   - output/all_datasets_drift_results.csv  (full results)
#   - output/all_datasets_drift_deltas.csv   (drift changes)
#   - output/volatility_indicators.csv       (fluctuation/volatility metrics)
#   - output/regional_breakdown.csv          (eICU regional analysis)
#   - output/care_demographics_correlation.csv (care quartile vs demographics)
#   - figures/fig1-6*.png                    (main figures, 6 total per X2)
#   - output/review_response.pdf             (reviewer response document)
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
RUN_REVIEW=false
TOTAL_STEPS=8
BOOTSTRAP_ARGS=""
SOFA_THRESHOLD_ARGS=""

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            RUN_ANALYSIS=false
            RUN_FIGURES=false
            TOTAL_STEPS=3
            shift
            ;;
        --analysis)
            RUN_SETUP=false
            RUN_FIGURES=false
            TOTAL_STEPS=1
            shift
            ;;
        --figures)
            RUN_SETUP=false
            RUN_ANALYSIS=false
            TOTAL_STEPS=1
            shift
            ;;
        --review)
            RUN_SETUP=false
            RUN_ANALYSIS=false
            RUN_FIGURES=false
            RUN_REVIEW=true
            TOTAL_STEPS=1
            shift
            ;;
        --fast)
            BOOTSTRAP_ARGS="--fast"
            echo -e "${YELLOW}FAST MODE: Using 2 bootstrap iterations (for testing)${NC}"
            shift
            ;;
        --bootstrap|-b)
            BOOTSTRAP_ARGS="--bootstrap $2"
            echo -e "${YELLOW}BOOTSTRAP: Using $2 iterations${NC}"
            shift 2
            ;;
        --sofa-thresholds)
            SOFA_THRESHOLD_ARGS="--sofa-thresholds $2"
            echo -e "${YELLOW}SOFA THRESHOLDS: $2${NC}"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./run_all.sh [--setup|--analysis|--figures|--review] [--fast|--bootstrap N] [--sofa-thresholds \"2 8 10\"]"
            exit 1
            ;;
    esac
done

STEP=0

# ============================================================
# REVIEW-ONLY MODE
# ============================================================
if [ "$RUN_REVIEW" = true ]; then
    STEP=$((STEP + 1))
    print_step $STEP "Generating Review Response Document"

    activate_venv

    echo "Creating review response document..."
    if [ -f "temporal/create_review_response.py" ]; then
        python temporal/create_review_response.py
    else
        echo "Review response script not found (responses consolidated in temporal/reviewer-responses.md)"
    fi

    echo ""
    echo -e "${GREEN}Review response document generated!${NC}"
    echo "  Output: output/review_response.pdf"
    echo ""
    exit 0
fi

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
        echo -e "${RED}ERROR: Python not found. Please install Python 3.10+${NC}"
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
    echo "  - Saltz ICU (27,259 patients)"
    echo "  - Zhejiang ICU (7,932 patients)"
    echo "  - eICU (289,503 patients)"
    echo "  - eICU-New (371,855 patients)"
    echo ""

    python code/batch_analysis.py $BOOTSTRAP_ARGS $SOFA_THRESHOLD_ARGS

    echo ""
    echo "MIMIC-IV Subsets (SOFA + Care Frequency):"
    echo "  - MIMIC-IV Mouthcare (8,675 patients)"
    echo "  - MIMIC-IV Mech. Vent. (8,919 patients)"
    echo ""

    python code/supplementary_analysis.py $BOOTSTRAP_ARGS

    # ============================================================
    # STEP 5: eICU Regional Analysis
    # ============================================================
    if [ -d "data/eicu" ] || [ -f "data/eicu.csv" ] || [ -f "data/eICU_data.csv" ]; then
        STEP=$((STEP + 1))
        print_step $STEP "Running eICU Regional Analysis"

        echo "Breaking down eICU results by region (Midwest, Northeast, South, West)..."
        python code/batch_analysis.py --eicu-regional $BOOTSTRAP_ARGS $SOFA_THRESHOLD_ARGS || echo -e "${YELLOW}eICU regional analysis encountered an issue (non-fatal)${NC}"

        echo ""
        echo -e "${GREEN}eICU regional analysis complete!${NC}"
    else
        echo ""
        echo -e "${YELLOW}Skipping eICU regional analysis (eICU data not found)${NC}"
    fi

    echo ""
    echo -e "${GREEN}Batch analysis complete!${NC}"
fi

# ============================================================
# STEP 6: Generate Figures
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

    # ============================================================
    # STEP 7: Generate Review Response Document
    # ============================================================
    STEP=$((STEP + 1))
    print_step $STEP "Generating Review Response Document"

    echo "Creating review response document..."
    if [ -f "temporal/create_review_response.py" ]; then
        python temporal/create_review_response.py
    else
        echo "Review response script not found (responses consolidated in temporal/reviewer-responses.md)"
    fi

    echo ""
    echo -e "${GREEN}Review response document generated!${NC}"
fi

# ============================================================
# STEP 8: Summary
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
    for f in figures/fig*.png figures/fig*.svg; do
        [ -f "$f" ] && echo "    - $(basename $f)"
    done
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PIPELINE COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Key outputs:"
echo "  - Full Results:       output/all_datasets_drift_results.csv"
echo "  - Drift Deltas:       output/all_datasets_drift_deltas.csv"
echo "  - Volatility:         output/volatility_indicators.csv"
echo "  - Regional Breakdown: output/regional_breakdown.csv"
echo "  - Care Demographics:  output/care_demographics_correlation.csv"
echo "  - Main Figures:       figures/fig2_gender_race.png, fig3_age_race.png, fig4_gender_age_race.png, fig5_mouthcare.png, fig6_mechvent.png"
echo "  - Review Response:    output/review_response.pdf"
echo ""
echo "To regenerate specific outputs:"
echo "  ./run_all.sh --analysis   # Rerun analysis"
echo "  ./run_all.sh --figures    # Regenerate figures"
echo "  ./run_all.sh --review     # Regenerate review response only"
echo ""
