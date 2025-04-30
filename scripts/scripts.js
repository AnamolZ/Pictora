document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            setActiveTab(tabId);
        });
    });

    function setActiveTab(tabId) {
        tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
        });
        tabPanes.forEach(pane => {
            pane.classList.toggle('active', pane.id === tabId);
        });
    }
});

window.onload = function () {
    const authSection = document.getElementById('auth-section');
    const userEmail = authSection.dataset.userEmail;
    const userName = authSection.dataset.userName;

    if (userEmail && userName) {
        localStorage.setItem('userEmail', userEmail);
        localStorage.setItem('userName', userName);
    }

    const cachedEmail = localStorage.getItem('userEmail');
    const cachedName = localStorage.getItem('userName');

    if (cachedEmail && cachedName) {
        authSection.innerHTML = `
            <a href="#" class="nav-link" style="color: black;">Models</a>
            <a href="#" id="payment-button" class="nav-link" style="color: black;">Payment</a>
            <a href="#" id="logout-link" class="nav-link" style="color: black;">
                ${cachedName}
            </a>
        `;
        document.getElementById('logout-link').addEventListener('click', function (e) {
            e.preventDefault();
            localStorage.clear();
            window.location.href = '/logout';
        });
        setupPayment();
    }
};

document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = uploadBox.querySelector('input[type="file"]');
    const captionTextDiv = uploadBox.querySelector('.caption-text');
    const captionEditor = uploadBox.querySelector('.caption-editor');
    const captionContainer = uploadBox.querySelector('.caption-container');
    const fileNameContainer = document.createElement('div');
    const previewImage = document.getElementById('previewImage');
    const processBtn = document.getElementById('processBtn');
    const clearBtn = document.getElementById('clearBtn');
    const editBtn = document.getElementById('editBtn');
    const defaultImage = 'https://images.pexels.com/photos/33041/antelope-canyon-lower-canyon-arizona.jpg';

    fileNameContainer.className = 'upload-name';
    uploadBox.appendChild(fileNameContainer);

    uploadBox.addEventListener('click', () => {
        if (!uploadBox.classList.contains('edit-mode')) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            showNotification("File must be an image type!");
            return;
        }

        const reader = new FileReader();
        reader.onload = function () {
            previewImage.src = reader.result;
        };
        reader.readAsDataURL(file);

        fileNameContainer.innerHTML = `<p class="upload-text">Selected: ${file.name}</p>`;
        fileNameContainer.classList.add('active');

        processBtn.disabled = false;
        editBtn.disabled = false;
    });

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!uploadBox.classList.contains('edit-mode')) {
            uploadBox.classList.add('drag-over');
        }
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');

        if (uploadBox.classList.contains('edit-mode')) return;

        const file = e.dataTransfer.files[0];
        if (!file) return;

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
    });

    clearBtn.addEventListener('click', () => {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        const fileNameContainers = document.querySelectorAll('.upload-name');

        fileInputs.forEach(input => input.value = '');
        fileNameContainers.forEach(container => {
            container.innerHTML = '';
            container.classList.remove('active');
        });

        previewImage.src = defaultImage;

        const uploadBox = document.getElementById('uploadBox');
        uploadBox.classList.remove('edit-mode');

        const captionContainer = uploadBox.querySelector('.caption-container');
        const captionEditor = uploadBox.querySelector('.caption-editor');

        if (captionContainer) {
            captionContainer.style.display = 'block';
            captionContainer.querySelector('.upload-text').textContent = 'Click to upload an image';
            captionContainer.querySelector('.upload-limit').textContent = 'or drag and drop';
        }

        if (captionEditor) {
            captionEditor.style.display = 'none';
        }

        processBtn.disabled = true;
        editBtn.disabled = true;
    });

    processBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        const selectedLanguage = document.getElementById('language').value;
        const userToken = localStorage.getItem('userToken');

        if (!file) {
            captionEditor.style.display = 'block';
            captionContainer.style.display = 'none';
            captionTextDiv.textContent = 'Please select an image before processing.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', selectedLanguage);
        formData.append('token', userToken);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                credentials: 'include',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                captionEditor.style.display = 'block';
                captionContainer.style.display = 'none';
                captionTextDiv.textContent = result.caption || 'No caption returned';
                uploadBox.classList.add('edit-mode');

                if (result.message) {
                    showNotification(result.message);
                }
            } else {
                captionEditor.style.display = 'block';
                captionContainer.style.display = 'none';
                captionTextDiv.textContent = result.caption || '';
                showNotification(result.message || 'An unknown error occurred.');
            }
        } catch (err) {
            captionEditor.style.display = 'block';
            captionContainer.style.display = 'none';
            captionTextDiv.textContent = 'An error occurred while sending the image.';
        }
    });

    editBtn.addEventListener('click', () => {
        if (!uploadBox.classList.contains('edit-mode')) return;

        const isEditing = captionTextDiv.isContentEditable;

        if (!isEditing) {
            captionTextDiv.contentEditable = 'true';
            captionTextDiv.focus();
            editBtn.textContent = 'Lock';
            editBtn.classList.add('active');
        } else {
            captionTextDiv.contentEditable = 'false';
            editBtn.textContent = 'Edit';
            editBtn.classList.remove('active');
        }
    });

    const copyBtn = document.querySelector('.copy-btn');
    const captionText = document.querySelector('.caption-text');

    if (copyBtn && captionText) {
        copyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const captionContent = captionText.textContent;

            navigator.clipboard.writeText(captionContent).then(() => {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 1500);
            }).catch(err => {
                console.error('Could not copy text: ', err);
            });
        });
    }
});

function toggleApiKey() {
    const token = localStorage.getItem('userToken');
    const apiKeyElement = document.getElementById('api-key');
    const button = document.getElementById('api-key-button');
    isLogin = localStorage.getItem('userEmail');

    if (!isLogin) {
        showNotification("Login To Get Access To API Key")
    }

    if (apiKeyElement.textContent === '********************************************') {
        if (token) {
            apiKeyElement.textContent = token;
            button.textContent = 'Hide API Key';
        }
    } else {
        apiKeyElement.textContent = '********************************************';
        button.textContent = 'Show API Key';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const copyBtn = document.querySelector('.copy-btn1');
    const apiKeyElement = document.getElementById('api-key');

    if (copyBtn && apiKeyElement) {
        copyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const tokenKey  = apiKeyElement.textContent;

            navigator.clipboard.writeText(tokenKey).then(() => {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 1500);
            }).catch(err => {
                console.error('Could not copy text: ', err);
            });
        });
    }
});

function showNotification(message, duration = 5000) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.classList.remove('hidden');

    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.classList.add('hidden');
        }, 400);
    }, duration);
}

document.addEventListener('DOMContentLoaded', () => {
    const zipInput = document.getElementById('zipInput');
    const zipUploadBox = document.getElementById('zipUploadBox');
    const zipProcessBtn = document.getElementById('zipProcessBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const languageSelect = document.getElementById('language_bp');

    let selectedZipFile = null;
    let downloadUrl = null;

    zipUploadBox.addEventListener('click', () => {
        zipInput.click();
    });

    zipInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.size > 5 * 1024 * 1024) {
            showNotification('File too large. Max size is 5MB.');
            zipInput.value = '';
            selectedZipFile = null;
            return;
        }

        selectedZipFile = file;
        zipUploadBox.querySelector('.upload-text').textContent = `Selected: ${file.name}`;
        zipUploadBox.classList.add('active');
    });

    zipProcessBtn.addEventListener('click', async () => {
        if (!selectedZipFile) {
            showNotification('Please select a ZIP file.');
            return;
        }

        const selectedLanguage = languageSelect.value;

        zipProcessBtn.disabled = true;
        zipProcessBtn.textContent = 'Processing...';
        downloadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedZipFile);
        formData.append('language', selectedLanguage);

        try {
            const response = await fetch('/batchprocessor', {
                method: 'POST',
                body: formData,
                credentials: 'include',
            });

            if (response.ok) {
                const blob = await response.blob();
                downloadUrl = window.URL.createObjectURL(blob);

                downloadBtn.disabled = false;

                downloadBtn.addEventListener('click', function handleClick() {
                    const link = document.createElement('a');
                    link.href = downloadUrl;
                    link.download = 'captions.json';
                    link.click();
                    downloadBtn.removeEventListener('click', handleClick);
                });

                showNotification('ZIP processed successfully! Click to download the captions.');
            } else {
                const error = await response.json();
                showNotification(error.detail || 'Error processing ZIP file.');
            }
        } catch (err) {
            showNotification('Failed to process the ZIP file.');
        } finally {
            zipProcessBtn.disabled = false;
            zipProcessBtn.textContent = 'Process Zip File';
        }
    });
});

document.addEventListener('DOMContentLoaded', () => {
    setupPayment();
});

function setupPayment() {
    const paymentButton = document.getElementById('payment-button');
    const paymentPage = document.getElementById('payment-page');
    const closePayment = document.getElementById('close-payment');

    if (!paymentButton || !paymentPage) return;

    paymentButton.addEventListener('click', (e) => {
        e.preventDefault();
        paymentPage.classList.add('active');
        document.body.style.overflow = 'hidden';
    });

    closePayment?.addEventListener('click', () => {
        paymentPage.classList.remove('active');
        document.body.style.overflow = '';
    });

    paymentPage?.addEventListener('click', (e) => {
        if (e.target === paymentPage) {
            paymentPage.classList.remove('active');
            document.body.style.overflow = '';
        }
    });

    const planButtons = document.querySelectorAll('.select-plan-btn');
    planButtons.forEach(button => {
        button.addEventListener('click', () => {
            const price = button.closest('.pricing-card')?.dataset.price;
            const isLoggedIn = localStorage.getItem('userEmail');
            if (isLoggedIn) {
                if (price) {
                    window.location.href = `/create_payment?amount=${encodeURIComponent(price)}.00`;
                }
            } else {
                window.location.href = '/login';
            }
        });
    });
}
