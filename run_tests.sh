#!/bin/bash
cd "$(dirname "$0")"
echo "========================================"
echo "  Auto Grading System - Unit Tests"
echo "========================================"
echo ""
python3 -X utf8 -m pytest tests/ -v --tb=short
