import { setupImageUpload } from './imageUpload.js';
import { setupCaptionGenerator } from './captionGenerator.js';
import { setupUI } from './ui.js';

document.addEventListener('DOMContentLoaded', () => {
  setupTheme();
  setupUI();
  setupImageUpload();
  setupCaptionGenerator();
  setupPayment();
});

window.onload = function () {
  const authSection = document.getElementById('auth-section');

  const userEmail = authSection.dataset.userEmail;
  const userPhoto = authSection.dataset.userPhoto;

  if (userEmail && userPhoto) {
    localStorage.setItem('userEmail', userEmail);
    localStorage.setItem('userPhoto', userPhoto);
  }

  const cachedEmail = localStorage.getItem('userEmail');
  const cachedPhoto = localStorage.getItem('userPhoto');

  if (cachedEmail && cachedPhoto) {
    authSection.innerHTML = `
      <div id="profile-picture-container">
        <img src="${cachedPhoto}" alt="Profile" width="40" height="40" style="border-radius: 50%;"/>
      </div>
      <a href="#" id="logout-link" style="display: inline-block; margin-top: 4px; font-size: 12px; color: white; text-decoration: none;">Logout</a>
    `;

    document.getElementById('logout-link').addEventListener('click', function (e) {
      e.preventDefault();
      localStorage.clear();
      window.location.href = '/';
    });
  }
}

function setupTheme() {
  const themeButton = document.getElementById('theme-button');
  const storedTheme = localStorage.getItem('theme');

  if (storedTheme === 'light') {
    document.body.classList.add('light-theme');
  }

  themeButton.addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
    localStorage.setItem('theme', currentTheme);
    createRipple(themeButton);
  });
}

function setupPayment() {
  const paymentButton = document.getElementById('payment-button');
  const paymentPage = document.getElementById('payment-page');
  const closePayment = document.getElementById('close-payment');

  paymentButton.addEventListener('click', () => {
    paymentPage.classList.add('active');
    document.body.style.overflow = 'hidden';
    createRipple(paymentButton);
  });

  closePayment.addEventListener('click', () => {
    paymentPage.classList.remove('active');
    document.body.style.overflow = '';
  });

  paymentPage.addEventListener('click', (e) => {
    if (e.target === paymentPage) {
      paymentPage.classList.remove('active');
      document.body.style.overflow = '';
    }
  });

  const planButtons = document.querySelectorAll('.select-plan-btn');
  planButtons.forEach(button => {
    button.addEventListener('click', () => {
      createRipple(button);
      const price = button.closest('.pricing-card').dataset.price;
      window.location.href = `/create_payment?amount=${encodeURIComponent(price)}.00`;
    });
  });  
}

function createRipple(button) {
  const ripple = document.createElement('span');
  const rect = button.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const x = event.clientX - rect.left - size / 2;
  const y = event.clientY - rect.top - size / 2;

  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${x}px`;
  ripple.style.top = `${y}px`;
  ripple.classList.add('ripple');

  button.appendChild(ripple);

  setTimeout(() => {
    ripple.remove();
  }, 600);
}

document.querySelectorAll('.generate-btn, .action-btn').forEach(button => {
  button.addEventListener('click', function() {
    if (!this.classList.contains('loading') && !this.disabled) {
      createRipple(this);
    }
  });
});