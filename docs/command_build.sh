#!/bin/bash
# Build documentation using datasci environment

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate datasci

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import pydata_sphinx_theme; print('✓ pydata-sphinx-theme installed')" || {
    echo "Installing pydata-sphinx-theme..."
    pip install pydata-sphinx-theme
}

python -c "import sphinx_design; print('✓ sphinx-design installed')" || {
    echo "Installing sphinx-design..."
    pip install sphinx-design
}

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/

# Generate API documentation
echo "Generating API documentation..."
sphinx-autogen source/index.rst

# Build HTML documentation
echo "Building HTML documentation..."
sphinx-build -M html source build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✓ Documentation built successfully!"
    echo "Open build/html/index.html to view the documentation"
else
    echo "✗ Documentation build failed!"
    exit 1
fi