export function setupUI() {
    document.title = 'Pictora Image Captioner';
    addButtonHoverEffects();
    setupInitialStates();
  }
  
  function addButtonHoverEffects() {
    const buttons = document.querySelectorAll('button:not([disabled])');
  
    buttons.forEach(button => {
      button.addEventListener('mouseover', () => {
        button.style.transform = 'translateY(-2px)';
        button.style.transition = 'transform 0.2s ease';
      });
  
      button.addEventListener('mouseout', () => {
        button.style.transform = 'translateY(0)';
      });
    });
  }
  
  function setupInitialStates() {
    const loadingSpinner = document.querySelector('.loading-spinner');
    if (loadingSpinner) {
      loadingSpinner.classList.add('hidden');
    }
  
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) {
      generateBtn.disabled = true;
    }
  
    const primaryColor = getComputedStyle(document.documentElement)
      .getPropertyValue('--color-primary')
      .trim();
  
    if (primaryColor.startsWith('#')) {
      const r = parseInt(primaryColor.slice(1, 3), 16);
      const g = parseInt(primaryColor.slice(3, 5), 16);
      const b = parseInt(primaryColor.slice(5, 7), 16);
      document.documentElement.style.setProperty('--color-primary-rgb', `${r}, ${g}, ${b}`);
    }
  }
  
  export function isTouchDevice() {
    return ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
  }
  
  export function hexToRgb(hex) {
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, (m, r, g, b) => r + r + g + g + b + b);
  
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16)
        }
      : null;
  }