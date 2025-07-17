// Animations and Interactivity for the Landing Page

document.addEventListener('DOMContentLoaded', function () {
    // Theme Toggle Functionality
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');

    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            document.body.classList.toggle('dark-mode');
            if (document.body.classList.contains('dark-mode')) {
                themeIcon.textContent = '☀️';
                localStorage.setItem('theme', 'dark');
            } else {
                themeIcon.textContent = '🌙';
                localStorage.setItem('theme', 'light');
            }
        });

        // Check for saved theme preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeIcon.textContent = '☀️';
        }
    }

    // Scroll Reveal Animation
    const revealElements = document.querySelectorAll('.reveal-on-scroll');

    function checkScroll() {
        revealElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;

            if (elementTop < windowHeight - 100) {
                const delay = element.getAttribute('data-delay') || 0;
                const animation = element.getAttribute('data-animation') || 'animate__fadeIn';

                setTimeout(() => {
                    element.classList.add('is-visible');
                    element.classList.add('animate__animated');
                    element.classList.add(animation);
                }, delay * 1000);
            }
        });
    }

    window.addEventListener('scroll', checkScroll);
    // Check initially when page loads
    checkScroll();

    // Counter Animation
    const counterElements = document.querySelectorAll('.counter-value');

    function startCounters() {
        counterElements.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'), 10) || 0;
            const duration = 2000; // 2 seconds
            const step = target / (duration / 16); // 60fps

            let current = 0;
            const counterInterval = setInterval(() => {
                current += step;
                if (current >= target) {
                    counter.textContent = target.toLocaleString();
                    clearInterval(counterInterval);
                } else {
                    counter.textContent = Math.floor(current).toLocaleString();
                }
            }, 16);
        });
    }

    // Start counters when they come into view
    const statsSection = document.getElementById('stats');
    if (statsSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    startCounters();
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });

        observer.observe(statsSection);
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Account for fixed navbar
                    behavior: 'smooth'
                });
            }
        });
    });

    // Mobile Navigation
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarNav = document.querySelector('#navbarNav');

    if (navbarToggler && navbarNav) {
        document.querySelectorAll('#navbarNav .nav-link').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < 992) { // Bootstrap lg breakpoint
                    const bsCollapse = new bootstrap.Collapse(navbarNav);
                    bsCollapse.hide();
                }
            });
        });
    }

    // Add animation delay to hero elements
    document.querySelectorAll('.hero-section .animate__animated').forEach((el, index) => {
        const delay = el.getAttribute('data-delay') || (0.1 * index);
        el.style.animationDelay = `${delay}s`;
    });
});
