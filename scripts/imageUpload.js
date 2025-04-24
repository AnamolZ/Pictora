export function setupImageUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadContainer = document.getElementById('upload-container');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const changeImageBtn = document.getElementById('change-image-btn');
    const generateBtn = document.getElementById('generate-btn');
  
    uploadArea.addEventListener('click', () => {
      fileInput.click();
    });
  
    fileInput.addEventListener('change', handleFileSelection);
  
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });
  
    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });
  
    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
  
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelection();
      }
    });
  
    changeImageBtn.addEventListener('click', () => {
      imagePreviewContainer.classList.add('hidden');
      uploadContainer.classList.remove('hidden');
      fileInput.value = '';
      generateBtn.disabled = true;
    });
  
    function handleFileSelection() {
      const file = fileInput.files[0];
  
      if (file) {
        if (!file.type.startsWith('image/')) {
          showError('Please select a valid image file');
          return;
        }
  
        const reader = new FileReader();
  
        reader.onload = (e) => {
          imagePreview.src = e.target.result;
          uploadContainer.classList.add('hidden');
          imagePreviewContainer.classList.remove('hidden');
          imagePreviewContainer.classList.add('fade-in');
          generateBtn.disabled = false;
        };
  
        reader.onerror = () => {
          showError('Error reading the image file');
        };
  
        reader.readAsDataURL(file);
      }
    }
  
    function showError(message) {
      alert(message);
    }
  
    function getUploadedImage() {
      return fileInput.files[0] || null;
    }
  
    return {
      getUploadedImage
    };
  }