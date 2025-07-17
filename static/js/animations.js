/**
 * animations.js - Handles animations and interactions for the AI Interview Question Generator
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize animations for elements with animate__animated classes
    initAnimations();

    // Initialize counters for statistics
    initCounters();

    // Initialize scroll reveal animations
    initScrollReveal();
});

/**
 * Initialize animations for elements that should animate on page load
 */
function initAnimations() {
    // Animate elements with delay
    document.querySelectorAll('.animate__animated').forEach(element => {
        // If element has a data-delay attribute, set the animation delay
        if (element.dataset.delay) {
            element.style.animationDelay = element.dataset.delay + 's';
        }
    });
}

/**
 * Initialize counter animations for statistics
 */
function initCounters() {
    const counters = document.querySelectorAll('.counter-value');

    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 2000; // 2 seconds
        const step = target / (duration / 16); // Update every 16ms (60fps)
        let current = 0;

        const updateCounter = () => {
            current += step;
            if (current < target) {
                counter.textContent = Math.ceil(current).toLocaleString();
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target.toLocaleString();
            }
        };

        // Start the counter animation when the element is in viewport
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    updateCounter();
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        observer.observe(counter);
    });
}

/**
 * Initialize scroll reveal animations for elements
 */
function initScrollReveal() {
    const revealElements = document.querySelectorAll('.reveal-on-scroll');

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated');
                entry.target.classList.add(entry.target.dataset.animation || 'animate__fadeIn');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    revealElements.forEach(element => {
        revealObserver.observe(element);
    });
}

/**
 * Show content based on tab selection
 * @param {Event} event - The click event
 * @param {string} tabId - The ID of the tab to show
 */
function showTab(event, tabId) {
    event.preventDefault();

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('d-none');
    });

    // Deactivate all tabs
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show the selected tab content
    document.getElementById(tabId).classList.remove('d-none');

    // Activate the clicked tab
    event.currentTarget.classList.add('active');
}

/**
 * Update theme icon based on current theme
 * @param {string} theme - The current theme ('light' or 'dark')
 */
function updateThemeIcon(theme) {
    const themeIcon = document.getElementById('themeIcon');
    if (themeIcon) {
        themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}
