# IASC Project Blog

This directory contains a Python script to generate an attractive project webpage showcasing the IASC system and all generated language handbooks.

## Features

- **Modern Design**: Styled similar to Sakana.ai blog pages with gradients and smooth interactions
- **Responsive Layout**: Works on desktop and mobile devices
- **Collapsible Handbooks**: Each language handbook can be expanded/collapsed for easy browsing
- **Structured Content**: Sections like PHONOLOGY, ORTHOGRAPHY, MORPHOSYNTAX, etc. are nicely formatted
- **Rich Formatting**:
  - Section headers stand out with colored styling
  - Lists (numbered and bulleted) are properly formatted
  - Example texts are displayed in monospace font with special styling
  - IPA symbols and linguistic glosses are preserved

## Usage

### Generate the webpage

```bash
python3 blog/build_blog.py
```

This will:
1. Parse all handbook files from the `handbooks/` directory
2. Extract intro text from `README.md`
3. Generate `blog/index.html` with all content

### View the webpage

Open the generated file in your browser:

```bash
open blog/index.html  # macOS
xdg-open blog/index.html  # Linux
start blog/index.html  # Windows
```

Or use the convenience script:

```bash
./blog/view.sh
```

## Output

The generated `index.html` contains:

1. **Header**: IASC logo and title
2. **Links**: GitHub repository and arXiv paper buttons
3. **System Diagram**: Placeholder for future diagram
4. **Introduction**: Text from README.md explaining ConLangs
5. **Handbooks**: All language handbooks in collapsible sections, showing:
   - Language name (from handbook title)
   - Syntax source language (e.g., Arabic, Turkish)
   - Phonology source language (e.g., Welsh, Japanese)
   - Writing script (e.g., Cyrillic, Latin, Arabic, Greek)
   - Full handbook content with proper formatting

## Files

- `build_blog.py` - Main Python script to generate the HTML
- `index.html` - Generated output (created by running the script)
- `instructions.txt` - Original requirements for the blog
- `README.md` - This file

## Customization

Edit `build_blog.py` to customize:
- Color scheme (CSS variables)
- Layout and spacing
- Section formatting rules
- Font choices
