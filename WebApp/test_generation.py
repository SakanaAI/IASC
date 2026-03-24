#!/usr/bin/env python3
"""Test script for phonotactics generation - helps debug issues."""

import os
import sys
import subprocess
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_generation():
    """Test the phonotactics generation process."""
    print("=" * 60)
    print("Testing IASC Phonotactics Generation")
    print("=" * 60)
    print()

    # Create temp directory
    output_dir = os.path.join(tempfile.gettempdir(), 'iasc_test')
    os.makedirs(output_dir, exist_ok=True)
    phonotactics_base = os.path.join(output_dir, 'phonotactics')

    print(f"Output directory: {output_dir}")
    print()

    # Build command
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cmd = [
        sys.executable,
        os.path.join(project_root, 'agentic_phonology', 'run_phonology_main.py'),
        '--which_task=phonotactics',
        '--language=Japanese',
        f'--phonotactics_base={phonotactics_base}',
        '--max_iter=2',  # Just 2 iterations for testing
        '--model=claude',
        '--num_output_examples=20',
        '--num_closest=1',
        '--user_prompt_dump',
    ]

    print("Command:")
    print(' '.join(cmd))
    print()

    # Check if AWS credentials are set
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        print("WARNING: AWS_ACCESS_KEY_ID not set!")
    if not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("WARNING: AWS_SECRET_ACCESS_KEY not set!")
    print()

    # Run the command
    print("Running phonotactics generation...")
    print("-" * 60)

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout for testing
    )

    print(f"Return code: {result.returncode}")
    print()

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
        print()

    if result.stderr:
        print("STDERR:")
        print(result.stderr)
        print()

    # Check what files were created
    print("-" * 60)
    print("Generated files:")
    try:
        files = sorted(os.listdir(output_dir))
        for f in files:
            file_path = os.path.join(output_dir, f)
            size = os.path.getsize(file_path)
            print(f"  {f} ({size} bytes)")

        # Try to run the final phonotactics file
        py_files = [f for f in files if f.endswith('.py')]
        if py_files:
            last_py = sorted(py_files)[-1]
            print()
            print(f"Testing {last_py}:")
            test_result = subprocess.run(
                [sys.executable, os.path.join(output_dir, last_py), '--num_morphemes', '5'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if test_result.returncode == 0:
                print("Sample words:")
                print(test_result.stdout)
            else:
                print(f"Error running file: {test_result.stderr}")

    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=" * 60)
    print(f"Test complete. Output in: {output_dir}")
    print("=" * 60)

    return result.returncode == 0


if __name__ == '__main__':
    success = test_generation()
    sys.exit(0 if success else 1)
