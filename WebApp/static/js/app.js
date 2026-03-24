// IASC Phonotactics Generator - Frontend JavaScript

let currentSessionId = null;
let currentIteration = 0;
let lastIterationNumber = -1;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Number input controls
function incrementIterations() {
    const input = document.getElementById('iterations');
    const currentValue = parseInt(input.value) || 10;
    if (currentValue < 50) {
        input.value = currentValue + 1;
    }
}

function decrementIterations() {
    const input = document.getElementById('iterations');
    const currentValue = parseInt(input.value) || 10;
    if (currentValue > 1) {
        input.value = currentValue - 1;
    }
}

function setupEventListeners() {
    // Toggle API keys panel
    document.getElementById('toggleApiKeys').addEventListener('click', function() {
        const panel = document.getElementById('apiKeysPanel');
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';

        // Update API key visibility when opening the panel
        if (panel.style.display !== 'none') {
            updateApiKeyVisibility();
        }
    });

    // Model selection change - update visible API keys
    document.getElementById('model').addEventListener('change', function() {
        updateApiKeyVisibility();
    });

    // Initialize API key visibility
    updateApiKeyVisibility();

    // Generate button
    document.getElementById('generateBtn').addEventListener('click', function() {
        generatePhonotactics();
    });

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', function() {
        if (currentSessionId) {
            downloadFinalGenerator();
        }
    });
}

function updateApiKeyVisibility() {
    const model = document.getElementById('model').value;

    // Hide all key groups
    document.getElementById('claudeKeys').style.display = 'none';
    document.getElementById('openaiKeys').style.display = 'none';
    document.getElementById('geminiKeys').style.display = 'none';

    // Show relevant key group
    if (model.includes('claude')) {
        document.getElementById('claudeKeys').style.display = 'block';
    } else if (model.includes('gpt')) {
        document.getElementById('openaiKeys').style.display = 'block';
    } else if (model.includes('gemini')) {
        document.getElementById('geminiKeys').style.display = 'block';
    }
}

async function generatePhonotactics() {
    // Collect form data
    const language = document.getElementById('language').value;
    const iterations = parseInt(document.getElementById('iterations').value);
    const model = document.getElementById('model').value;

    // Collect API keys (if provided)
    const awsAccessKey = document.getElementById('awsAccessKey').value;
    const awsSecretKey = document.getElementById('awsSecretKey').value;
    const openaiKey = document.getElementById('openaiKey').value;
    const geminiKey = document.getElementById('geminiKey').value;

    // Validate
    if (iterations < 1 || iterations > 50) {
        showError('Please enter a number of iterations between 1 and 50');
        return;
    }

    // Build request payload
    const payload = {
        language: language,
        iterations: iterations,
        model: model
    };

    // Add API keys if provided
    if (awsAccessKey) payload.aws_access_key_id = awsAccessKey;
    if (awsSecretKey) payload.aws_secret_access_key = awsSecretKey;
    if (openaiKey) payload.openai_api_key = openaiKey;
    if (geminiKey) payload.gemini_api_key = geminiKey;

    // Show progress UI
    showProgress();

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Generation failed');
        }

        // Store session ID
        currentSessionId = data.session_id;

        // Display results
        displayResults(data.results);

    } catch (error) {
        console.error('Error:', error);
        showError(`Generation failed: ${error.message}`);
    } finally {
        hideProgress();
    }
}

function showProgress() {
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('progressArea').style.display = 'block';
    document.getElementById('resultsPanel').style.display = 'none';
}

function hideProgress() {
    document.getElementById('generateBtn').disabled = false;
    document.getElementById('progressArea').style.display = 'none';
}

function showError(message) {
    hideProgress();

    // Create error message element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <strong>Error:</strong>
        ${message}
    `;

    // Insert after generate button
    const btn = document.getElementById('generateBtn');
    btn.parentNode.insertBefore(errorDiv, btn.nextSibling);

    // Auto-remove after 10 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 10000);
}

function displayResults(results) {
    // Show results panel
    document.getElementById('resultsPanel').style.display = 'block';

    // Scroll to results
    document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth' });

    // Display phonemes
    displayPhonemes(results.final_phonemes);

    // Display sample words
    displaySampleWords(results.sample_words);

    // Display reasoning/iteration info
    displayIterationTabs(results.reasoning);
}

function displayPhonemes(phonemes) {
    // Display consonants
    const consonantsTbody = document.getElementById('consonants');
    consonantsTbody.innerHTML = '';

    if (phonemes.consonants && Object.keys(phonemes.consonants).length > 0) {
        // Sort by frequency (descending)
        const sortedConsonants = Object.entries(phonemes.consonants)
            .sort((a, b) => b[1] - a[1]);

        sortedConsonants.forEach(([symbol, frequency]) => {
            const row = createPhonemeRow(symbol, frequency);
            consonantsTbody.appendChild(row);
        });
    } else {
        consonantsTbody.innerHTML = '<tr><td colspan="2" class="phoneme-table-empty">No consonants found</td></tr>';
    }

    // Display vowels
    const vowelsTbody = document.getElementById('vowels');
    vowelsTbody.innerHTML = '';

    if (phonemes.vowels && Object.keys(phonemes.vowels).length > 0) {
        // Sort by frequency (descending)
        const sortedVowels = Object.entries(phonemes.vowels)
            .sort((a, b) => b[1] - a[1]);

        sortedVowels.forEach(([symbol, frequency]) => {
            const row = createPhonemeRow(symbol, frequency);
            vowelsTbody.appendChild(row);
        });
    } else {
        vowelsTbody.innerHTML = '<tr><td colspan="2" class="phoneme-table-empty">No vowels found</td></tr>';
    }
}

function createPhonemeRow(symbol, frequency) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>${escapeHtml(symbol)}</td>
        <td>${frequency}</td>
    `;
    return tr;
}

function displaySampleWords(words) {
    const wordsDiv = document.getElementById('sampleWords');
    wordsDiv.innerHTML = '';

    if (words && words.length > 0) {
        words.forEach(word => {
            const wordItem = document.createElement('div');
            wordItem.className = 'word-item';
            wordItem.textContent = word;
            wordsDiv.appendChild(wordItem);
        });
    } else {
        wordsDiv.innerHTML = '<p>No sample words generated</p>';
    }
}

function displayIterationTabs(reasoning) {
    const tabButtons = document.getElementById('iterationTabs');
    const notesContainer = document.getElementById('iterationNotes');
    const codeContainer = document.getElementById('iterationCode');

    tabButtons.innerHTML = '';
    notesContainer.innerHTML = '';
    codeContainer.innerHTML = '';

    if (!reasoning || reasoning.length === 0) {
        notesContainer.innerHTML = '<p>No reasoning information available</p>';
        codeContainer.innerHTML = '<p>No code generated</p>';
        lastIterationNumber = -1;
        return;
    }

    // Store the last iteration number for downloads
    lastIterationNumber = reasoning[reasoning.length - 1].iteration;

    // Create tab buttons and content for each iteration
    reasoning.forEach((item, index) => {
        // Create tab button
        const button = document.createElement('button');
        button.className = 'tab-button' + (index === 0 ? ' active' : '');
        button.textContent = `Iteration ${item.iteration}`;
        button.addEventListener('click', () => switchTab(item.iteration));
        tabButtons.appendChild(button);

        // Split content into notes and code
        const { notes, code } = splitOutputContent(item.content);

        // Create notes content
        const notesDiv = document.createElement('div');
        notesDiv.className = 'iteration-content' + (index === 0 ? ' active' : '');
        notesDiv.id = `notes-${item.iteration}`;
        notesDiv.textContent = notes || 'No additional notes for this iteration';
        notesContainer.appendChild(notesDiv);

        // Create code content
        const codeDiv = document.createElement('div');
        codeDiv.className = 'iteration-content' + (index === 0 ? ' active' : '');
        codeDiv.id = `code-${item.iteration}`;
        codeDiv.textContent = code || 'No code generated for this iteration';
        codeContainer.appendChild(codeDiv);
    });
}

function splitOutputContent(content) {
    // Split content by <OUTPUT> tags
    const outputMatch = content.match(/<OUTPUT>([\s\S]*?)<\/OUTPUT>/);

    let notes = '';
    let code = '';

    if (outputMatch) {
        // Everything before <OUTPUT> is notes
        const beforeOutput = content.split('<OUTPUT>')[0];
        notes = beforeOutput.trim();

        // Everything inside <OUTPUT></OUTPUT> is code
        code = outputMatch[1].trim();

        // Remove any markdown code fences if present
        if (code.startsWith('```python')) {
            code = code.replace(/```python\n?/, '').replace(/```\s*$/, '');
        } else if (code.startsWith('```')) {
            code = code.replace(/```\n?/, '').replace(/```\s*$/, '');
        }
    } else {
        // No OUTPUT tags found, treat entire content as notes
        notes = content;
    }

    return { notes, code };
}

function switchTab(iteration) {
    // Update button states
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach((button, index) => {
        if (button.textContent === `Iteration ${iteration}`) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });

    // Update notes content visibility
    const notesContents = document.querySelectorAll('#iterationNotes .iteration-content');
    notesContents.forEach(content => {
        if (content.id === `notes-${iteration}`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });

    // Update code content visibility
    const codeContents = document.querySelectorAll('#iterationCode .iteration-content');
    codeContents.forEach(content => {
        if (content.id === `code-${iteration}`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

async function downloadFinalGenerator() {
    if (!currentSessionId) {
        showError('No session available for download');
        return;
    }

    if (lastIterationNumber < 0) {
        showError('No iterations available for download');
        return;
    }

    try {
        // Build filename using the stored last iteration number
        const filename = `phonotactics_${String(lastIterationNumber).padStart(2, '0')}.py`;

        // Download the file
        window.location.href = `/api/download/${currentSessionId}/${filename}`;

    } catch (error) {
        console.error('Download error:', error);
        showError('Failed to download file');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Cleanup session when page is closed (optional)
window.addEventListener('beforeunload', function() {
    if (currentSessionId) {
        // Send beacon to cleanup (won't block page unload)
        navigator.sendBeacon(`/api/cleanup/${currentSessionId}`, '');
    }
});
