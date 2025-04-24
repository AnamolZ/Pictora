export function setupCaptionGenerator() {
  const generateBtn = document.getElementById('generate-btn');
  const captionsList = document.getElementById('captions-list');
  const resultsSection = document.getElementById('results-section');
  const copyAllBtn = document.getElementById('copy-all-btn');
  const downloadBtn = document.getElementById('download-btn');
  const languageSelect = document.getElementById('language-select');

  generateBtn.addEventListener('click', generateCaptions);
  copyAllBtn.addEventListener('click', copyAllCaptions);
  downloadBtn.addEventListener('click', downloadCaptions);

  async function generateCaptions() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const selectedLanguage = languageSelect.value;
    if (!file) return;

    setLoading(true);

    try {
      const captions = await uploadToFastAPI(file, selectedLanguage);
      displayCaptions(captions);

      resultsSection.classList.remove('hidden');
      resultsSection.classList.add('slide-in-up');
      resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
      showMessage(error.message);

    } finally {
      setLoading(false);
    }
  }

  async function uploadToFastAPI(file) {
    const formData = new FormData();
    formData.append('file', file);
    const selectedLanguage = languageSelect.value;
    formData.append('language', selectedLanguage);

    const response = await fetch('/process', {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error('Error processing the image, please try again later.');
    }

    if (result.message) {
      showMessage(result.message);
    }

    let captionText = result.caption || 'Error: No caption generated.';

    return [{
      text: captionText,
      confidence: 1.0
    }];
  }

  function displayCaptions(captions) {
    captionsList.innerHTML = '';

    captions.forEach((caption, index) => {
      const captionItem = document.createElement('div');
      captionItem.className = 'caption-item';
      captionItem.style.setProperty('--index', index);

      const confidencePercentage = Math.round(caption.confidence * 100);

      captionItem.innerHTML = `
        <p class="caption-text">${caption.text}</p>
        <div class="caption-meta">
          <span>Confidence: ${confidencePercentage}%</span>
          <button class="text-button copy-btn" data-text="${caption.text}">Copy</button>
        </div>
      `;

      captionsList.appendChild(captionItem);

      const copyBtn = captionItem.querySelector('.copy-btn');
      copyBtn.addEventListener('click', () => {
        copyToClipboard(caption.text);
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copy-success');
        setTimeout(() => {
          copyBtn.textContent = 'Copy';
          copyBtn.classList.remove('copy-success');
        }, 1500);
      });
    });
  }

  function setLoading(isLoading) {
    const loadingSpinner = document.querySelector('.loading-spinner');

    if (isLoading) {
      generateBtn.classList.add('loading');
      loadingSpinner.classList.remove('hidden');
    } else {
      generateBtn.classList.remove('loading');
      loadingSpinner.classList.add('hidden');
    }
  }

  function copyAllCaptions() {
    const captions = Array.from(document.querySelectorAll('.caption-text'))
      .map(el => el.textContent)
      .join('\n\n');

    copyToClipboard(captions);

    copyAllBtn.classList.add('copy-success');
    const originalText = copyAllBtn.innerHTML;
    copyAllBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
      </svg>
      Copied!
    `;

    setTimeout(() => {
      copyAllBtn.classList.remove('copy-success');
      copyAllBtn.innerHTML = originalText;
    }, 1500);
  }

  function downloadCaptions() {
    const captions = Array.from(document.querySelectorAll('.caption-text'))
      .map(el => el.textContent)
      .join('\n\n');

    const blob = new Blob([captions], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'image-captions.txt';
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }

  function copyToClipboard(text) {
    navigator.clipboard.writeText(text).catch(err => {
      console.error('Failed to copy text: ', err);
    });
  }

  function showMessage(message) {
    alert(message);
  }
}