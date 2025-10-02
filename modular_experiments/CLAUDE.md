# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a modular experiments repository for constructed language (ConLang) generation using large language models. The system generates phonology, orthography, and morphosyntax for artificial languages based on real-world language features.

## Architecture

### Core Components

- **Runner Scripts** (`run_*.py`): Main orchestration scripts for different language aspects
  - `run_phonology.py` - Generates phonotactics based on existing languages
  - `run_orthography.py` - Creates writing systems based on phonology and scripts  
  - `run_morphosyntax.py` - Develops grammar systems with configurable morphosyntactic features
  - `run_corpus_creation.py` - Creates text corpora for generated languages
  - `run_handbook.py` - Generates language handbooks/documentation

- **Parameter System** (`morphosyntax_params.py`): Pydantic-based models defining morphosyntactic features
  - `Morphosyntax` class combining syntax and morphology parameters
  - Pre-defined language configurations (Turkish, French, Arabic, Welsh, Vietnamese, Mizo, Fijian, Hixkaryana)
  - Supports typologically diverse features (word order, case systems, tense/aspect, etc.)

### Execution Flow

1. **Phonology Stage**: Creates phonotactics based on source languages (French, Hawaiian, Japanese, Spanish, Welsh)
2. **Orthography Stage**: Develops writing systems using various scripts (Latin, Cyrillic, Arabic, Greek)
3. **Morphosyntax Stage**: Iteratively builds grammar through multiple stages:
   - Word order (SOV, SVO, VSO, etc.)
   - Morphological features (case, tense, person agreement, etc.)
   - Advanced features (voice, mood, relativization, etc.)

### Output Structure

Generated outputs are organized under `modular_experiment_outputs/` with subdirectories by:
- Model type (claude, gpt-4o-mini, gemini-2.5-flash, etc.)
- Language configuration (turkish, french, arabic, etc.)
- Experiment stage and iteration numbers

## Common Commands

### Running Experiments

```bash
# Generate phonology for a language
python3 run_phonology.py --languages="Japanese" --model="claude" --run_commands

# Generate morphosyntax with specific parameters
python3 run_morphosyntax.py --model="claude" --premade_params_language="turkish" --num_iter=3 --reference_file="path/to/reference.txt" --run_commands

# Run orthography experiments
python3 run_orthography.py --languages="Japanese" --scripts="Latin" --run_commands

# Create corpus for generated language
python3 run_corpus_creation.py --model="claude" --run_commands

# Generate language handbook
python3 run_handbook.py --model="claude" --run_commands
```

### Using Shell Scripts

Pre-configured shell scripts are available for common workflows:
```bash
# Run phonology experiments
./controlled_phonology_claude.sh

# Run morphosyntax for all languages
./controlled_morphosyntax/claude/all_languages.sh

# Run specific language configurations
./controlled_morphosyntax/claude/turkish.sh
```

### Evaluation

The morphosyntax pipeline includes evaluation capabilities:
```bash
python3 run_morphosyntax.py --pipeline --reference_file="reference_glosses.txt" --model="claude" --premade_params_language="turkish"
```

This will generate evaluation scripts and run morphosyntactic analysis comparing model outputs to reference glosses.

## Development Notes

### Adding New Languages

To add support for a new language configuration:

1. Create a new function in `morphosyntax_params.py` following the pattern `sample_params_LANGUAGE()`
2. Define the `Morphosyntax` object with appropriate `Syntax` and `Morphology` parameters
3. Add the function to the `LANGUAGE_TO_PARAMS` dictionary
4. Update command-line argument choices in relevant runner scripts

### Model Integration

The system supports multiple LLM backends through the `--model` parameter:
- `claude` - Default Claude model
- `gpt-4o-mini` - OpenAI GPT-4o mini
- `gemini-2.5-flash` - Google Gemini 2.5 Flash
- Custom models can be added by updating model choices in argument parsers

### Parameter Customization

The Pydantic-based parameter system allows for:
- Type-safe morphosyntactic feature definitions
- Hierarchical organization of linguistic features
- Easy extension for new morphological categories
- Validation of parameter combinations

Key parameter categories include:
- **Syntax**: Word order patterns and alignment systems
- **Case**: Marking strategies and case inventories  
- **Tense/Aspect**: Temporal and aspectual distinctions
- **Agreement**: Person/number marking on verbs
- **Morphology Type**: Isolating, agglutinative, or fusional