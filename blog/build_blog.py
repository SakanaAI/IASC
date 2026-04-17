#!/usr/bin/env python3
"""
Build an attractive project webpage for IASC with handbooks displayed
in collapsible sections, styled similar to Sakana.ai blogs.
"""

import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple


def image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64 data URI"""
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Determine the image type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    }.get(ext, 'image/png')

    # Encode to base64
    encoded = base64.b64encode(image_data).decode('utf-8')

    return f'data:{mime_type};base64,{encoded}'


def extract_intro_text(readme_path: str) -> str:
    """Extract the intro text from README.md"""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract text from "# IASC: Interactive Agentic System for ConLangs" section
    # until the next # heading
    pattern = r'# IASC: Interactive Agentic System for ConLangs\n\n(.*?)\n\n#'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def parse_handbook_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse filename like 'Arabic_Welsh_Cyrillic.txt' to extract:
    - syntax_source: Arabic
    - phonology_source: Welsh
    - script: Cyrillic
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')

    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def parse_handbook(filepath: str) -> Dict[str, any]:
    """Parse a handbook file and extract structured content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract title
    title_match = re.search(r'TITLE:\s*(.+)', content)
    language_name = title_match.group(1).replace('A Grammar and Handbook of ', '') if title_match else "Unknown Language"

    # Extract sections
    sections = {}

    # Split by section headers (all caps words at start of line)
    section_pattern = r'^([A-Z\s]+)\n\n'
    parts = re.split(section_pattern, content, flags=re.MULTILINE)

    current_section = None
    for i, part in enumerate(parts):
        if i == 0:
            continue  # Skip content before first section

        if part.strip().isupper() and part.strip() not in ['TITLE']:
            current_section = part.strip()
        elif current_section:
            sections[current_section] = part.strip()

    return {
        'language_name': language_name,
        'sections': sections
    }


def format_handbook_section(section_name: str, content: str) -> str:
    """Format a handbook section with proper HTML"""
    html = f'<h3 class="section-header">{section_name}</h3>\n'

    # Split content into paragraphs
    paragraphs = content.split('\n\n')

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1
            continue

        # Check if it's a numbered list with glossed examples
        # These have the pattern: number. text\n   gloss\n   "translation"
        if re.match(r'^\d+\.', para) and section_name == 'EXAMPLE TEXT':
            # Collect all consecutive numbered examples
            examples = []
            while i < len(paragraphs):
                para = paragraphs[i].strip()
                if re.match(r'^\d+\.', para):
                    examples.append(para)
                    i += 1
                elif para and not re.match(r'^\d+\.', para):
                    # Not a numbered item anymore, break
                    break
                else:
                    i += 1

            # Format as numbered list of examples
            html += '<ol class="example-list">\n'
            for example in examples:
                # Remove the number prefix
                example_text = re.sub(r'^\d+\.\s*', '', example)
                # Split into lines (text, gloss, translation)
                lines = example_text.split('\n')
                html += '<li class="example-item">\n'
                for line in lines:
                    line = line.strip()
                    if line:
                        html += f'<div class="example-line">{line}</div>\n'
                html += '</li>\n'
            html += '</ol>\n'
            continue

        # Check if it's a regular numbered list
        elif re.match(r'^\d+\.', para):
            items = re.split(r'\n(?=\d+\.)', para)
            html += '<ol class="handbook-list">\n'
            for item in items:
                # Remove the number prefix
                item_text = re.sub(r'^\d+\.\s*', '', item)
                # Handle sub-items (lines starting with -)
                item_text = re.sub(r'\n\s*-\s*', '<br>&nbsp;&nbsp;&nbsp;• ', item_text)
                html += f'<li>{item_text}</li>\n'
            html += '</ol>\n'

        # Check if it's a bulleted list
        elif re.match(r'^[-•]', para):
            items = re.split(r'\n(?=[-•])', para)
            html += '<ul class="handbook-list">\n'
            for item in items:
                item_text = re.sub(r'^[-•]\s*', '', item)
                html += f'<li>{item_text}</li>\n'
            html += '</ul>\n'

        # Check if it's an example (contains IPA or linguistic glossing)
        elif re.search(r'[ɪəɛɔʃʒθðŋχɬ]', para) or '"' in para:
            # Split by lines to preserve example formatting
            lines = para.split('\n')
            html += '<div class="example-text">\n'
            for line in lines:
                if line.strip():
                    html += f'<div class="example-line">{line}</div>\n'
            html += '</div>\n'

        # Regular paragraph
        else:
            # Preserve line breaks within paragraphs
            para_html = para.replace('\n', '<br>\n')
            html += f'<p class="handbook-paragraph">{para_html}</p>\n'

        i += 1

    return html


def generate_html(intro_text: str, handbooks: List[Dict], logo_data_uri: str, redfish_data_uri: str) -> str:
    """Generate the complete HTML page"""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IASC: Interactive Agentic System for ConLangs</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #fafbfc 0%, #f0f4f8 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        header {
            text-align: center;
            margin-bottom: 60px;
            padding: 40px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        }

        .logo {
            width: 150px;
            height: auto;
            margin-bottom: 20px;
        }

        .redfish-logo {
            width: 100px;
            height: auto;
            margin-bottom: 20px;
        }

        h1 {
            font-size: 2.5em;
            color: #1a1a1a;
            margin-bottom: 20px;
            font-weight: 700;
        }

        .authors {
            display: flex;
            gap: 60px;
            justify-content: center;
            margin: 25px 0;
            flex-wrap: wrap;
        }

        .author {
            text-align: center;
        }

        .author-name {
            font-size: 1.1em;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .author-name a {
            color: #1a1a1a;
            text-decoration: none;
            transition: color 0.2s;
        }

        .author-name a:hover {
            color: #667eea;
        }

        .author-affiliation {
            font-size: 0.9em;
            color: #666;
        }

        .links {
            margin-top: 25px;
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .links a {
            display: inline-flex;
            align-items: center;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .links a:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .intro {
            background: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 40px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.08);
            line-height: 1.8;
        }

        .intro p {
            margin-bottom: 15px;
            color: #444;
        }

        .diagram-placeholder {
            background: linear-gradient(135deg, #e0e7ff 0%, #cfe0ff 100%);
            padding: 80px 40px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 40px;
            border: 2px dashed #667eea;
        }

        .diagram-placeholder p {
            color: #667eea;
            font-size: 1.2em;
            font-weight: 500;
        }

        .handbooks-section {
            margin-top: 60px;
        }

        .handbooks-section > h2 {
            font-size: 2em;
            margin-bottom: 30px;
            color: #1a1a1a;
            text-align: center;
        }

        .handbook {
            background: white;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            transition: box-shadow 0.3s;
        }

        .handbook:hover {
            box-shadow: 0 4px 25px rgba(0,0,0,0.12);
        }

        .handbook-header {
            padding: 25px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }

        /* Rainbow colors for handbook headers */
        .handbook:nth-child(7n+1) .handbook-header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); /* Pink-Red */
        }
        .handbook:nth-child(7n+2) .handbook-header {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); /* Pink-Yellow */
        }
        .handbook:nth-child(7n+3) .handbook-header {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); /* Yellow-Blue */
        }
        .handbook:nth-child(7n+4) .handbook-header {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); /* Cyan-Pink */
        }
        .handbook:nth-child(7n+5) .handbook-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Purple */
        }
        .handbook:nth-child(7n+6) .handbook-header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); /* Blue-Cyan */
        }
        .handbook:nth-child(7n+7) .handbook-header {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); /* Green-Cyan */
        }

        .handbook-header:hover {
            opacity: 0.9;
        }

        .handbook-title {
            font-size: 1.4em;
            font-weight: 600;
        }

        .handbook-meta {
            font-size: 0.9em;
            opacity: 0.95;
            margin-top: 5px;
        }

        .toggle-icon {
            font-size: 1.5em;
            transition: transform 0.3s;
        }

        .handbook-header.active .toggle-icon {
            transform: rotate(180deg);
        }

        .handbook-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out;
        }

        .handbook-content.active {
            max-height: 10000px;
            transition: max-height 0.6s ease-in;
        }

        .handbook-inner {
            padding: 40px;
        }

        .section-header {
            font-size: 1.5em;
            color: #667eea;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e7ff;
            font-weight: 600;
        }

        .section-header:first-child {
            margin-top: 0;
        }

        .handbook-paragraph {
            margin-bottom: 15px;
            color: #444;
            line-height: 1.8;
        }

        .handbook-list {
            margin: 15px 0 15px 30px;
            color: #444;
        }

        .handbook-list li {
            margin-bottom: 8px;
            line-height: 1.7;
        }

        .example-text {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
        }

        .example-line {
            margin-bottom: 8px;
            line-height: 1.6;
        }

        .example-list {
            margin: 20px 0 20px 30px;
            counter-reset: example-counter;
            list-style: none;
        }

        .example-list .example-item {
            counter-increment: example-counter;
            margin-bottom: 25px;
            position: relative;
            padding-left: 10px;
        }

        .example-list .example-item::before {
            content: counter(example-counter) ".";
            position: absolute;
            left: -25px;
            font-weight: 600;
            color: #667eea;
        }

        .example-list .example-item .example-line {
            background: #f8f9fa;
            padding: 8px 15px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            line-height: 1.6;
        }

        .example-list .example-item .example-line:first-child {
            font-weight: 500;
            background: #e8ecff;
        }

        .example-list .example-item .example-line:nth-child(2) {
            color: #666;
            font-size: 0.95em;
        }

        .example-list .example-item .example-line:nth-child(3) {
            font-style: italic;
            color: #444;
        }

        footer {
            text-align: center;
            padding: 40px 20px;
            color: #666;
            margin-top: 60px;
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 1.8em;
            }

            .handbook-title {
                font-size: 1.1em;
            }

            .links {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <img src="{logo_data_uri}" alt="IASC Logo" class="logo">
            <img src="{redfish_data_uri}" alt="Redfish Logo" class="redfish-logo">
            <h1>IASC: Interactive Agentic System for ConLangs</h1>
            <div class="authors">
                <div class="author">
                    <div class="author-name"><a href="https://ctaguchi.github.io/" target="_blank">Chihiro Taguchi</a></div>
                    <div class="author-affiliation">Sakana AI, Notre Dame University</div>
                </div>
                <div class="author">
                    <div class="author-name"><a href="https://rws.xoba.com" target="_blank">Richard Sproat</a></div>
                    <div class="author-affiliation">Sakana AI</div>
                </div>
            </div>
            <div class="links">
                <a href="https://github.com/SakanaAI/IASC/" target="_blank">📦 GitHub Repository</a>
                <a href="https://arxiv.org/abs/2510.07591" target="_blank">📄 arXiv Paper</a>
            </div>
        </header>

        <div class="diagram-placeholder">
            <p>🔧 System Diagram Placeholder</p>
        </div>

        <div class="intro">
"""

    # Add intro paragraphs
    for para in intro_text.split('\n\n'):
        if para.strip():
            # Convert markdown links [text](url) to HTML <a href="url">text</a>
            para_html = re.sub(
                r'\[([^\]]+)\]\(([^\)]+)\)',
                r'<a href="\2" target="_blank">\1</a>',
                para.strip()
            )
            html += f"            <p>{para_html}</p>\n"

    html += """        </div>

        <div class="handbooks-section">
            <h2>Generated Language Handbooks</h2>
"""

    # Add each handbook
    for hb in handbooks:
        html += f"""
            <div class="handbook">
                <div class="handbook-header" onclick="toggleHandbook(this)">
                    <div>
                        <div class="handbook-title">{hb['language_name']}</div>
                        <div class="handbook-meta">Syntax: {hb['syntax']} | Phonology: {hb['phonology']} | Script: {hb['script']}</div>
                    </div>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="handbook-content">
                    <div class="handbook-inner">
"""

        # Add sections
        for section_name, content in hb['sections'].items():
            html += format_handbook_section(section_name, content)

        html += """                    </div>
                </div>
            </div>
"""

    html += """        </div>

        <footer>
            <p>© 2024 Sakana AI. Generated with IASC.</p>
        </footer>
    </div>

    <script>
        function toggleHandbook(header) {
            header.classList.toggle('active');
            const content = header.nextElementSibling;
            content.classList.toggle('active');
        }
    </script>
</body>
</html>
"""

    # Replace the logo placeholders with the actual data URIs
    html = html.replace('{logo_data_uri}', logo_data_uri)
    html = html.replace('{redfish_data_uri}', redfish_data_uri)

    return html


def main():
    # Paths
    repo_root = Path(__file__).parent.parent
    handbooks_dir = repo_root / 'handbooks'
    readme_path = repo_root / 'README.md'
    logo_path = repo_root / 'iasc.png'
    redfish_path = Path(__file__).parent / 'redfish.png'
    output_path = Path(__file__).parent / 'index.html'

    # Convert logos to base64 data URIs
    print("Embedding logos...")
    logo_data_uri = image_to_base64(str(logo_path))
    redfish_data_uri = image_to_base64(str(redfish_path))

    # Extract intro text
    intro_text = extract_intro_text(str(readme_path))

    # Parse all handbooks
    handbooks = []
    for filepath in sorted(handbooks_dir.glob('*.txt')):
        syntax, phonology, script = parse_handbook_filename(filepath.name)
        handbook_data = parse_handbook(str(filepath))
        handbook_data['syntax'] = syntax
        handbook_data['phonology'] = phonology
        handbook_data['script'] = script
        handbook_data['filename'] = filepath.name
        handbooks.append(handbook_data)

    print(f"Found {len(handbooks)} handbooks")

    # Generate HTML
    html = generate_html(intro_text, handbooks, logo_data_uri, redfish_data_uri)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Generated {output_path}")
    print(f"  Open file://{output_path.absolute()} in your browser")


if __name__ == '__main__':
    main()
