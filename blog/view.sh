#!/bin/bash
# Convenience script to generate and view the blog

cd "$(dirname "$0")/.."

echo "Generating blog..."
python3 blog/build_blog.py

echo ""
echo "Opening in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open blog/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open blog/index.html
else
    echo "Please open blog/index.html in your browser"
fi
