// Fix for tab navigation issue in app.js
document.addEventListener('DOMContentLoaded', function() {
    console.log("Tab fix script loaded");
    
    // Wait a short time for all other scripts to initialize
    setTimeout(function() {
        // Override the existing openTab function
        window.openTab = function(evt, tabName) {
            console.log("Fixed openTab function called for tab:", tabName);
            
            // Get tab id either from parameter or from event
            const tabId = typeof tabName === 'string' ? tabName : evt.currentTarget.getAttribute('data-tab');
            if (!tabId) {
                console.error("No tab ID found");
                return;
            }
            
            // Hide all tab content immediately (no fade out)
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = 'none';
                tabContents[i].classList.remove('active');
            }
            
            // Remove active class from all tab buttons
            const tabButtons = document.getElementsByClassName('tab-btn');
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].classList.remove('active');
            }
            
            // Show the selected tab immediately
            const selectedTab = document.getElementById(tabId);
            if (selectedTab) {
                selectedTab.classList.add('active');
                selectedTab.style.display = 'block';
                selectedTab.style.opacity = '1';
                console.log("Tab activated:", tabId);
            }
            
            // Add active class to clicked button
            if (evt && evt.currentTarget) {
                evt.currentTarget.classList.add('active');
            }
        };
        
        // Add data-tab attributes to tab buttons if missing
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabIds = ['performance', 'features', 'visualizations', 'advanced-viz', 'user-analytics'];
        
        tabButtons.forEach((button, index) => {
            if (!button.hasAttribute('data-tab') && index < tabIds.length) {
                button.setAttribute('data-tab', tabIds[index]);
            }
        });
        
        // Set up click handlers for all tab buttons
        document.querySelectorAll('.tab-btn').forEach(button => {
            // Remove existing click event listeners
            const newButton = button.cloneNode(true);
            if (button.parentNode) {
                button.parentNode.replaceChild(newButton, button);
            }
            
            // Add our fixed click handler
            newButton.addEventListener('click', function(event) {
                event.preventDefault();
                window.openTab(event, newButton.getAttribute('data-tab'));
            });
        });
        
        // Set up image modal functionality
        setupImageModal();
    }, 500);
    
    // Set up image modal functionality
    function setupImageModal() {
        // Create modal if it doesn't exist
        if (!document.getElementById('image-modal')) {
            const modal = document.createElement('div');
            modal.id = 'image-modal';
            modal.className = 'modal';
            modal.innerHTML = `
                <span class="modal-close">&times;</span>
                <img class="modal-content" id="modal-image">
                <div class="modal-caption" id="modal-caption"></div>
            `;
            document.body.appendChild(modal);
            
            // Close button functionality
            modal.querySelector('.modal-close').addEventListener('click', function() {
                modal.style.display = 'none';
            });
            
            // Close on click outside
            window.addEventListener('click', function(event) {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
            
            // Close on Escape key
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape' && modal.style.display === 'flex') {
                    modal.style.display = 'none';
                }
            });
        }
        
        // Add click event to all plot images
        document.querySelectorAll('.plot img').forEach(img => {
            img.style.cursor = 'pointer';
            img.addEventListener('click', function() {
                const modal = document.getElementById('image-modal');
                const modalImg = document.getElementById('modal-image');
                const captionText = document.getElementById('modal-caption');
                
                modal.style.display = 'flex';
                modalImg.src = this.src;
                
                // Get caption from nearest h3
                const plotContainer = this.closest('.plot');
                const heading = plotContainer ? plotContainer.querySelector('h3') : null;
                captionText.textContent = heading ? heading.textContent : 'Visualization';
            });
        });
    }
});