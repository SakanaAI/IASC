"""Flask web application for IASC Phonotactics Generator."""
import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Import IASC modules for parsing only (not for direct execution)
try:
    from agentic_phonology import loader
except ImportError:
    loader = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Supported languages
SUPPORTED_LANGUAGES = ["French", "Hawaiian", "Japanese", "Spanish", "Welsh"]

# Store generation sessions in memory (in production, use Redis or similar)
generation_sessions = {}


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)


@app.route('/api/generate', methods=['POST'])
def generate_phonotactics():
    """
    Generate phonotactics based on user input.

    Expected JSON payload:
    {
        "language": "Japanese",
        "iterations": 10,
        "model": "claude",
        "aws_access_key_id": "...",  # Optional, can use env vars
        "aws_secret_access_key": "...",  # Optional, can use env vars
        "openai_api_key": "...",  # Optional for GPT models
        "gemini_api_key": "..."  # Optional for Gemini models
    }
    """
    try:
        data = request.get_json()

        # Validate input
        language = data.get('language')
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Unsupported language: {language}'}), 400

        iterations = int(data.get('iterations', 10))
        if iterations < 1 or iterations > 50:
            return jsonify({'error': 'Iterations must be between 1 and 50'}), 400

        model = data.get('model', 'claude')

        # Create temporary directory for outputs
        session_id = os.urandom(16).hex()
        output_dir = os.path.join(tempfile.gettempdir(), f'iasc_phonology_{session_id}')
        os.makedirs(output_dir, exist_ok=True)

        phonotactics_base = os.path.join(output_dir, 'phonotactics')

        # Set up environment for subprocess (API keys)
        env = os.environ.copy()
        if data.get('aws_access_key_id'):
            env['AWS_ACCESS_KEY_ID'] = data.get('aws_access_key_id')
        if data.get('aws_secret_access_key'):
            env['AWS_SECRET_ACCESS_KEY'] = data.get('aws_secret_access_key')
        if data.get('openai_api_key'):
            env['OPENAI_API_KEY'] = data.get('openai_api_key')
        if data.get('gemini_api_key'):
            env['GEMINI_API_KEY'] = data.get('gemini_api_key')

        # Run the phonotactics generation as a subprocess
        # This is necessary because the agentic_phonology code uses absl flags
        # which don't work well when called multiple times in the same process
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cmd = [
            sys.executable,
            os.path.join(project_root, 'agentic_phonology', 'run_phonology_main.py'),
            '--which_task=phonotactics',
            f'--language={language}',
            f'--phonotactics_base={phonotactics_base}',
            f'--max_iter={iterations}',
            f'--model={model}',
            '--num_output_examples=20',
            '--num_closest=1',
            '--user_prompt_dump',
        ]

        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        print(f"Output directory: {output_dir}")

        result = subprocess.run(
            cmd,
            env=env,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")

        if result.returncode != 0:
            error_msg = f"Phonotactics generation failed:\n{result.stderr}"
            if result.stdout:
                error_msg += f"\n\nOutput:\n{result.stdout}"
            raise Exception(error_msg)

        # Parse the results
        results = parse_phonotactics_results(output_dir, iterations)

        # Store session info
        generation_sessions[session_id] = {
            'output_dir': output_dir,
            'language': language,
            'results': results
        }

        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'results': results
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_phonotactics: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500


def parse_phonotactics_results(output_dir: str, iterations: int) -> Dict[str, Any]:
    """Parse the generated phonotactics files and extract key information."""
    results = {
        'iterations': [],
        'final_phonemes': {},
        'sample_words': [],
        'reasoning': []
    }

    print(f"Parsing results from: {output_dir}")
    print(f"Expected iterations: {iterations}")

    # List all files in output directory for debugging
    try:
        files = os.listdir(output_dir)
        print(f"Files in output directory: {files}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    # Find the last successful iteration
    last_iteration = -1
    for i in range(iterations):
        py_file = os.path.join(output_dir, f'phonotactics_{i:02d}.py')
        if os.path.exists(py_file):
            last_iteration = i
            print(f"Found iteration {i}")

    print(f"Last successful iteration: {last_iteration}")

    if last_iteration == -1:
        print("WARNING: No phonotactics files found!")
        return results

    # Parse each iteration's text file for reasoning
    for i in range(last_iteration + 1):
        txt_file = os.path.join(output_dir, f'phonotactics_{i:02d}.txt')
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                results['reasoning'].append({
                    'iteration': i,
                    'content': content
                })

    # Parse the final Python file
    final_py_file = os.path.join(output_dir, f'phonotactics_{last_iteration:02d}.py')
    if os.path.exists(final_py_file):
        with open(final_py_file, 'r', encoding='utf-8') as f:
            code = f.read()

            # Extract phonemes
            consonants = extract_phoneme_dict(code, 'consonants')
            vowels = extract_phoneme_dict(code, 'vowels')

            results['final_phonemes'] = {
                'consonants': consonants,
                'vowels': vowels
            }

            # Generate sample words using the final phonotactics
            # Run the Python file directly since the temp directory isn't in sys.path
            try:
                result = subprocess.run(
                    [sys.executable, final_py_file, '--num_morphemes', '20'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=output_dir
                )
                if result.returncode == 0:
                    # Split output and filter out empty lines
                    words = [w.strip() for w in result.stdout.strip().split('\n') if w.strip()]
                    results['sample_words'] = words
                else:
                    print(f"Sample word generation failed: {result.stderr}")
                    results['sample_words'] = []
            except Exception as e:
                print(f"Error generating sample words: {e}")
                results['sample_words'] = []

    return results


def extract_phoneme_dict(code: str, var_name: str) -> Dict[str, int]:
    """Extract a phoneme dictionary from Python code.

    This handles both simple cases like:
        consonants = {"k": 10, "s": 9}
    And complex cases where consonants are built from multiple sub-dicts:
        plosives = {"p": 5, "t": 8}
        fricatives = {"f": 3, "s": 7}
        consonants = {**plosives, **fricatives}
    """
    phonemes = {}

    # First, try the simple case - direct dictionary assignment
    pattern = rf'{var_name}\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, code, re.DOTALL)

    if match:
        dict_str = match.group(1)
        # Check if it's a merge operation (contains **)
        if '**' not in dict_str:
            # Simple dictionary, parse it
            # Strategy: Process line by line, stripping comments first,
            # then extract all key:value pairs
            cleaned_lines = []
            for line in dict_str.split('\n'):
                # Remove everything after # (comments)
                if '#' in line:
                    line = line.split('#')[0]
                cleaned_lines.append(line)

            # Join back and now split by comma safely
            cleaned_text = ' '.join(cleaned_lines)

            # Extract all "key": value pairs using regex
            # This handles both "key": value and 'key': value
            pair_pattern = r'["\']([^"\']+)["\']\s*:\s*(\d+)'
            for match in re.finditer(pair_pattern, cleaned_text):
                key = match.group(1)
                value = int(match.group(2))
                phonemes[key] = value

            return phonemes

    # If we didn't find a simple dictionary, look for sub-dictionaries being merged
    # Common patterns: plosives, fricatives, affricates, nasals, etc. for consonants
    # Common patterns: front_vowels, back_vowels, etc. for vowels
    sub_dict_patterns = [
        r'(\w+)\s*=\s*\{([^}]+)\}',  # Find all dictionary definitions
    ]

    all_dicts = {}
    for pattern in sub_dict_patterns:
        for match in re.finditer(pattern, code):
            dict_name = match.group(1)
            dict_content = match.group(2)

            # Parse this dictionary using the same approach
            cleaned_lines = []
            for line in dict_content.split('\n'):
                if '#' in line:
                    line = line.split('#')[0]
                cleaned_lines.append(line)

            cleaned_text = ' '.join(cleaned_lines)

            temp_dict = {}
            pair_pattern = r'["\']([^"\']+)["\']\s*:\s*(\d+)'
            for pair_match in re.finditer(pair_pattern, cleaned_text):
                key = pair_match.group(1)
                value = int(pair_match.group(2))
                temp_dict[key] = value

            if temp_dict:
                all_dicts[dict_name] = temp_dict

    # Now look for the target variable being constructed from sub-dicts
    # Pattern like: consonants = {**plosives, **fricatives, ...}
    merge_pattern = rf'{var_name}\s*=\s*\{{([^}}]+)\}}'
    match = re.search(merge_pattern, code, re.DOTALL)

    if match:
        merge_content = match.group(1)
        # Extract all **dict_name references
        for sub_dict_ref in re.finditer(r'\*\*(\w+)', merge_content):
            sub_dict_name = sub_dict_ref.group(1)
            if sub_dict_name in all_dicts:
                phonemes.update(all_dicts[sub_dict_name])

    # If we still don't have phonemes and the target var_name is a simple dict we found
    if not phonemes and var_name in all_dicts:
        phonemes = all_dicts[var_name]

    return phonemes


@app.route('/api/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Download a generated file."""
    if session_id not in generation_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = generation_sessions[session_id]
    file_path = os.path.join(session['output_dir'], secure_filename(filename))

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)


@app.route('/api/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up temporary files for a session."""
    if session_id in generation_sessions:
        session = generation_sessions[session_id]
        output_dir = session['output_dir']

        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            del generation_sessions[session_id]
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Session not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
