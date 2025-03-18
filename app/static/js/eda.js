       // Reveal charts as user scrolls
       document.addEventListener("DOMContentLoaded", function() {
        const sections = document.querySelectorAll('.viz-section');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.15,
            rootMargin: '0px 0px -100px 0px'
        });
        
        sections.forEach(section => {
            observer.observe(section);
        });
        
        // Add image loading animation
        const images = document.querySelectorAll('.viz-container img');
        images.forEach(img => {
            img.style.opacity = '0';
            img.style.transition = 'opacity 0.5s ease';
            
            img.addEventListener('load', function() {
                this.style.opacity = '1';
            });
            
            // If image is already loaded
            if (img.complete) {
                img.style.opacity = '1';
            }
        });
    });