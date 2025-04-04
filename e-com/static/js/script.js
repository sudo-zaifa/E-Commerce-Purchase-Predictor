// E-Commerce Purchase Predictor - Enhanced JavaScript
// Current Date: 2025-04-03
// User: hzed120

// =========================================
// Global Variables and Initialization
// =========================================
document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let taskId = null;
    let intervalId = null;
    let modelData = null;
    let chartsInitialized = false;
    let darkMode = localStorage.getItem('darkMode') === 'enabled';
    
    // DOM Element References
    const elements = {
        // Main UI elements
        startButton: document.getElementById('start-pipeline'),
        progressFill: document.getElementById('progress-fill'),
        progressText: document.getElementById('progress-text'),
        statusMessage: document.getElementById('status-message'),
        resultsSection: document.getElementById('results-section'),
        predictBtn: document.getElementById('predict-btn'),
        randomSessionBtn: document.getElementById('random-session'),
        predictionResult: document.getElementById('prediction-result'),
        simulateSessionBtn: document.getElementById('simulate-session'),
        simulationPanel: document.getElementById('simulation-panel'),
        showDatasetInfoBtn: document.getElementById('show-dataset-info'),
        datasetInfoPanel: document.getElementById('dataset-info-panel'),
        downloadSampleBtn: document.getElementById('download-sample'),
        
        // Analytics and charts
        loadAdvancedBtn: document.getElementById('load-advanced-viz'),
        loadUserAnalyticsBtn: document.getElementById('load-user-analytics')
    };
    
    // Initialize timestamp updates
    initializeTimestamps();
    
    // Set up event listeners for all buttons and tabs
    setupEventListeners();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Create toast notification container
    createToastContainer();
    
    // =========================================
    // Core Functions
    // =========================================
    
    // Initialize timestamp displays with current time
    function initializeTimestamps() {
        updateTimestamps();
        setInterval(updateTimestamps, 1000);
        
        // Initialize formatted current time
        const now = new Date();
        document.getElementById('session-start-time').textContent = formatDate(now);
    }
    
    // Setup all event listeners
    function setupEventListeners() {
        // Training pipeline button
        if (elements.startButton) {
            elements.startButton.addEventListener('click', startPipeline);
        }
        
        // Prediction buttons
        if (elements.predictBtn) {
            elements.predictBtn.addEventListener('click', makePrediction);
        }
        
        if (elements.randomSessionBtn) {
            elements.randomSessionBtn.addEventListener('click', getRandomSession);
        }
        
        // Simulation buttons
        if (elements.simulateSessionBtn) {
            elements.simulateSessionBtn.addEventListener('click', showSimulationPanel);
        }
        
        const runSimulationBtn = document.getElementById('run-simulation');
        if (runSimulationBtn) {
            runSimulationBtn.addEventListener('click', runSessionSimulation);
        }
        
        // Analytics buttons
        if (elements.loadAdvancedBtn) {
            elements.loadAdvancedBtn.addEventListener('click', loadAdvancedVisualizations);
        }
        
        if (elements.loadUserAnalyticsBtn) {
            elements.loadUserAnalyticsBtn.addEventListener('click', loadUserAnalytics);
        }
        
        // Dataset buttons
        if (elements.showDatasetInfoBtn) {
            elements.showDatasetInfoBtn.addEventListener('click', toggleDatasetInfo);
        }
        
        if (elements.downloadSampleBtn) {
            elements.downloadSampleBtn.addEventListener('click', downloadSampleDataset);
        }
        
        // Setup plot click handlers for enlarging images
        setupPlotClickHandlers();
        
        // Setup tab navigation
        setupTabNavigation();
        
        // Add window resize listener for responsive charts
        window.addEventListener('resize', debounce(function() {
            if (chartsInitialized) {
                // Refresh charts when window size changes
                refreshChartsOnResize();
            }
        }, 250));
    }
    
    // Setup tab navigation
    function setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(button => {
            button.addEventListener('click', function(event) {
                openTab(event, this.getAttribute('data-tab') || this.textContent.toLowerCase().replace(/\s+/g, '-'));
            });
        });
    }
    
    // Setup click handlers for plots to show enlarged view
    function setupPlotClickHandlers() {
        document.addEventListener('click', function(e) {
            if (e.target && e.target.tagName === 'IMG' && e.target.closest('.plot')) {
                showImageModal(e.target.src);
            }
        });
    }
    
    // Initialize tooltips
    function initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(el => {
            const tooltip = document.createElement('span');
            tooltip.className = 'tooltip-text';
            tooltip.textContent = el.getAttribute('data-tooltip');
            el.classList.add('tooltip');
            el.appendChild(tooltip);
        });
    }
    
    // Create toast notification container
    function createToastContainer() {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Start the ML pipeline
    function startPipeline() {
        elements.startButton.disabled = true;
        elements.startButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Update status with typing effect
        typeWriterEffect(elements.statusMessage, 'Starting machine learning pipeline...', 50);
        
        elements.progressFill.style.width = '0%';
        elements.progressText.textContent = '0%';
        
        // Animate progress bar to 5% to indicate starting
        animateProgressTo(5);
        
        // Show notification
        showToast('info', 'Pipeline Started', 'The machine learning pipeline is starting up...', 5000);
        
        // Call the API to start the pipeline
        fetch('/start_pipeline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(handleResponse)
        .then(data => {
            taskId = data.task_id;
            checkStatus();
            intervalId = setInterval(checkStatus, 1000);
        })
        .catch(handleError);
    }
    
    // Check ML pipeline status
    function checkStatus() {
        if (!taskId) return;
        
        fetch(`/check_status/${taskId}`)
            .then(handleResponse)
            .then(data => {
                const progress = data.progress || 0;
                animateProgressTo(progress);
                
                if (data.status === 'completed') {
                    clearInterval(intervalId);
                    
                    // Update status with success message
                    elements.statusMessage.innerHTML = '<i class="fas fa-check-circle" style="color: var(--success-color);"></i> ML pipeline completed successfully!';
                    elements.startButton.disabled = false;
                    elements.startButton.innerHTML = '<i class="fas fa-check-circle"></i> Training Complete';
                    
                    // Show success notification
                    showToast('success', 'Pipeline Complete', 'Training completed successfully!', 5000);
                    
                    // Load results and plots with animation
                    setTimeout(() => {
                        loadResults();
                        setTimeout(loadPlots, 500);
                        elements.resultsSection.classList.remove('hidden');
                        fadeIn(elements.resultsSection);
                    }, 500);
                    
                } else if (data.status === 'failed') {
                    clearInterval(intervalId);
                    
                    // Update status with error message
                    elements.statusMessage.innerHTML = `<i class="fas fa-exclamation-triangle" style="color: var(--danger-color);"></i> Error: ${data.error || 'Pipeline failed'}`;
                    elements.startButton.disabled = false;
                    elements.startButton.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Training Failed';
                    
                    // Show error notification
                    showToast('error', 'Pipeline Failed', data.error || 'Processing failed', 8000);
                    
                } else {
                    // Update status message with current progress info
                    if (data.status) {
                        elements.statusMessage.innerHTML = `<i class="fas fa-cog fa-spin"></i> ${data.status}`;
                    } else {
                        elements.statusMessage.innerHTML = `<i class="fas fa-cog fa-spin"></i> Processing: ${progress}% complete`;
                    }
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                // Keep trying even if there's an error
            });
    }
    
    // Load model results
    function loadResults() {
        showLoading('results-table-body', 3);
        showLoading('feature-table-body', 5);
        
        fetch('/get_results')
            .then(handleResponse)
            .then(data => {
                modelData = data;
                
                // Update best model with animation
                const bestModelElement = document.getElementById('best-model');
                fadeTransition(bestModelElement, () => {
                    bestModelElement.textContent = data.best_model;
                });
                
                // Update results table
                const tableBody = document.getElementById('results-table-body');
                let tableHTML = '';
                
                for (const [name, result] of Object.entries(data.results)) {
                    const isHighlighted = name === data.best_model;
                    const rowClass = isHighlighted ? 'best-model' : '';
                    
                    const accuracy = (result.accuracy * 100).toFixed(2);
                    const f1Score = (result.f1_score * 100).toFixed(2);
                    const rocAuc = (result.roc_auc * 100).toFixed(2);
                    
                    tableHTML += `
                        <tr class="${rowClass}">
                            <td>${name}</td>
                            <td>${accuracy}%</td>
                            <td>${f1Score}%</td>
                            <td>${rocAuc}%</td>
                        </tr>
                    `;
                }
                fadeTransition(tableBody, () => {
                    tableBody.innerHTML = tableHTML;
                });
                
                // Update feature importance table
                const featureBody = document.getElementById('feature-table-body');
                let featureHTML = '';
                
                if (data.feature_importance && data.feature_importance.length > 0) {
                    // Get max importance for proper scaling
                    const maxImportance = data.feature_importance[0].importance;
                    
                    data.feature_importance.forEach((item, index) => {
                        const barWidth = (item.importance / maxImportance) * 100;
                        const delay = index * 100; // Stagger animation
                        
                        featureHTML += `
                            <tr>
                                <td>${item.feature}</td>
                                <td>${item.importance.toFixed(4)}</td>
                                <td>
                                    <div class="feature-bar-container">
                                        <div class="feature-bar" style="width: 0%" data-width="${barWidth}"></div>
                                    </div>
                                </td>
                            </tr>
                        `;
                    });
                }
                
                fadeTransition(featureBody, () => {
                    featureBody.innerHTML = featureHTML;
                    
                    // Animate feature bars
                    setTimeout(() => {
                        const bars = featureBody.querySelectorAll('.feature-bar');
                        bars.forEach((bar, index) => {
                            setTimeout(() => {
                                const targetWidth = bar.getAttribute('data-width') + '%';
                                bar.style.width = targetWidth;
                            }, index * 100);
                        });
                    }, 300);
                });
            })
            .catch(error => {
                console.error('Error loading results:', error);
                showToast('error', 'Error', 'Failed to load model results', 5000);
            });
    }
    
    // Load model plots
    function loadPlots() {
        // Show loading state for plots
        const plotIds = ['confusion-matrix', 'precision-recall', 'feature-importance', 'model-comparison'];
        plotIds.forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Cstyle%3E%40keyframes%20a%7B0%25%7Btransform%3Arotate%280%29%7Dto%7Btransform%3Arotate%28359deg%29%7D%7D%3C%2Fstyle%3E%3CcircleStyle%20fill%3D%22none%22%20stroke%3D%22%233b82f6%22%20stroke-width%3D%2210%22%20cx%3D%2250%22%20cy%3D%2250%22%20r%3D%2235%22%20stroke-dasharray%3D%22160%20100%22%20style%3D%22animation%3Aa%201s%20linear%20infinite%22%2F%3E%3C%2Fsvg%3E';
                img.alt = 'Loading...';
                img.classList.add('loading');
            }
        });
        
        fetch('/get_plots')
            .then(handleResponse)
            .then(data => {
                // Update plot images with fade-in effect
                plotIds.forEach(id => {
                    if (data[id]) {
                        const img = document.getElementById(id);
                        if (img) {
                            // Create new image off-screen to preload
                            const newImg = new Image();
                            newImg.onload = () => {
                                img.classList.remove('loading');
                                fadeTransition(img, () => {
                                    img.src = `data:image/png;base64,${data[id]}`;
                                    img.alt = id.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                });
                            };
                            newImg.src = `data:image/png;base64,${data[id]}`;
                        }
                    }
                });
                
                showToast('success', 'Visualizations Ready', 'Model visualizations have been loaded', 3000);
            })
            .catch(error => {
                console.error('Error loading plots:', error);
                showToast('error', 'Error', 'Failed to load visualizations', 5000);
                
                // Remove loading state
                plotIds.forEach(id => {
                    const img = document.getElementById(id);
                    if (img) {
                        img.classList.remove('loading');
                        img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Crect%20width%3D%22100%22%20height%3D%22100%22%20fill%3D%22%23f3f4f6%22%2F%3E%3Ctext%20x%3D%2250%22%20y%3D%2250%22%20font-family%3D%22sans-serif%22%20font-size%3D%2214%22%20text-anchor%3D%22middle%22%20dominant-baseline%3D%22middle%22%20fill%3D%22%236b7280%22%3EFailed%20to%20load%3C%2Ftext%3E%3C%2Fsvg%3E';
                        img.alt = 'Failed to load';
                    }
                });
            });
    }
    
    // Load advanced visualizations
    function loadAdvancedVisualizations() {
        const button = elements.loadAdvancedBtn;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        
        // List of advanced plots
        const plotIds = [
            'roc-curve', 'learning-curve', 'purchase-time-dist', 
            'purchase-device-dist', 'correlation-heatmap', 
            'user-flow', 'prob-distribution', 'cumulative-gains'
        ];
        
        // Show loading state
        plotIds.forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Cstyle%3E%40keyframes%20a%7B0%25%7Btransform%3Arotate%280%29%7Dto%7Btransform%3Arotate%28359deg%29%7D%7D%3C%2Fstyle%3E%3CcircleStyle%20fill%3D%22none%22%20stroke%3D%22%233b82f6%22%20stroke-width%3D%2210%22%20cx%3D%2250%22%20cy%3D%2250%22%20r%3D%2235%22%20stroke-dasharray%3D%22160%20100%22%20style%3D%22animation%3Aa%201s%20linear%20infinite%22%2F%3E%3C%2Fsvg%3E';
                img.alt = 'Loading...';
                img.classList.add('loading');
            }
        });
        
        fetch('/get_advanced_plots')
            .then(handleResponse)
            .then(data => {
                // Load each plot with a sequential delay for better UX
                plotIds.forEach((id, index) => {
                    if (data[id]) {
                        setTimeout(() => {
                            const img = document.getElementById(id);
                            if (img) {
                                // Create new image off-screen to preload
                                const newImg = new Image();
                                newImg.onload = () => {
                                    img.classList.remove('loading');
                                    fadeTransition(img, () => {
                                        img.src = `data:image/png;base64,${data[id]}`;
                                        img.alt = id.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                    });
                                };
                                newImg.src = `data:image/png;base64,${data[id]}`;
                            }
                        }, index * 200); // Stagger loading for visual appeal
                    }
                });
                
                // Re-enable button after all images are loaded
                setTimeout(() => {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-chart-bar"></i> Refresh Charts';
                    showToast('success', 'Advanced Charts', 'Advanced visualizations loaded', 3000);
                }, plotIds.length * 200 + 500);
            })
            .catch(error => {
                console.error('Error loading advanced plots:', error);
                showToast('error', 'Error', 'Failed to load advanced charts', 5000);
                
                // Remove loading state and show error images
                plotIds.forEach(id => {
                    const img = document.getElementById(id);
                    if (img) {
                        img.classList.remove('loading');
                        img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Crect%20width%3D%22100%22%20height%3D%22100%22%20fill%3D%22%23f3f4f6%22%2F%3E%3Ctext%20x%3D%2250%22%20y%3D%2250%22%20font-family%3D%22sans-serif%22%20font-size%3D%2214%22%20text-anchor%3D%22middle%22%20dominant-baseline%3D%22middle%22%20fill%3D%22%236b7280%22%3EFailed%20to%20load%3C%2Ftext%3E%3C%2Fsvg%3E';
                        img.alt = 'Failed to load';
                    }
                });
                
                // Re-enable button
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-chart-bar"></i> Try Again';
            });
    }
    
    // Load user analytics data and visualizations
    function loadUserAnalytics() {
        const button = elements.loadUserAnalyticsBtn;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        
        // Show loading in the summary container
        const summaryContainer = document.getElementById('user-analytics-summary');
        summaryContainer.innerHTML = `
            <div class="loading-container" style="text-align:center; padding:30px;">
                <i class="fas fa-spinner fa-spin" style="font-size:24px; margin-bottom:15px;"></i>
                <p>Loading user analytics data...</p>
            </div>
        `;
        
        // List of analytics plots
        const plotIds = [
            'device-distribution', 'hourly-activity', 'purchase-funnel', 
            'user-retention', 'category-performance'
        ];
        
        // Show loading state for plots
        plotIds.forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Cstyle%3E%40keyframes%20a%7B0%25%7Btransform%3Arotate%280%29%7Dto%7Btransform%3Arotate%28359deg%29%7D%7D%3C%2Fstyle%3E%3CcircleStyle%20fill%3D%22none%22%20stroke%3D%22%233b82f6%22%20stroke-width%3D%2210%22%20cx%3D%2250%22%20cy%3D%2250%22%20r%3D%2235%22%20stroke-dasharray%3D%22160%20100%22%20style%3D%22animation%3Aa%201s%20linear%20infinite%22%2F%3E%3C%2Fsvg%3E';
                img.alt = 'Loading...';
                img.classList.add('loading');
            }
        });
        
        fetch('/get_user_analytics')
            .then(handleResponse)
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update user analytics summary
                const summaryContainer = document.getElementById('user-analytics-summary');
                let summaryHTML = '<div class="summary-cards">';
                
                // Purchase metrics card
                if (data.purchase_metrics) {
                    const metrics = data.purchase_metrics;
                    summaryHTML += `
                        <div class="summary-card customers">
                            <div class="summary-icon">
                                <i class="fas fa-users"></i>
                            </div>
                            <div class="summary-data">
                                <div class="summary-value">${metrics.total_users}</div>
                                <div class="summary-label">Total Users</div>
                            </div>
                        </div>
                        
                        <div class="summary-card orders">
                            <div class="summary-icon">
                                <i class="fas fa-shopping-cart"></i>
                            </div>
                            <div class="summary-data">
                                <div class="summary-value">${metrics.purchasing_users}</div>
                                <div class="summary-label">Users with Purchases</div>
                            </div>
                        </div>
                        
                        <div class="summary-card conversion">
                            <div class="summary-icon">
                                <i class="fas fa-chart-line"></i>
                            </div>
                            <div class="summary-data">
                                <div class="summary-value">${metrics.conversion_rate}%</div>
                                <div class="summary-label">Conversion Rate</div>
                            </div>
                        </div>
                        
                        <div class="summary-card revenue">
                            <div class="summary-icon">
                                <i class="fas fa-check-circle"></i>
                            </div>
                            <div class="summary-data">
                                <div class="summary-value">${metrics.avg_purchases_per_buyer}</div>
                                <div class="summary-label">Avg. Purchases/Buyer</div>
                            </div>
                        </div>
                    `;
                }
                
                summaryHTML += '</div>';
                
                // Add purchase distribution if available
                if (data.purchase_distribution) {
                    summaryHTML += `
                        <div class="analytics-section">
                            <h3>Purchase Distribution</h3>
                            <div class="distribution-chart" id="purchase-distribution-chart"></div>
                        </div>
                    `;
                }
                
                // Add time analytics if available
                if (data.time_analytics) {
                    const peakHour = Object.entries(data.time_analytics.hourly_activity)
                        .sort((a, b) => b[1] - a[1])[0][0];
                    
                    const peakPurchaseHour = Object.entries(data.time_analytics.hourly_purchases || {})
                        .sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';
                    
                    summaryHTML += `
                        <div class="analytics-section">
                            <h3>Time-Based Insights</h3>
                            <div class="insights-grid">
                                <div class="insight-item">
                                    <div class="insight-icon"><i class="far fa-clock"></i></div>
                                    <div class="insight-content">
                                        <h4>Peak Activity Hour</h4>
                                        <div class="insight-value">${peakHour}:00</div>
                                    </div>
                                </div>
                                <div class="insight-item">
                                    <div class="insight-icon"><i class="fas fa-shopping-bag"></i></div>
                                    <div class="insight-content">
                                        <h4>Peak Purchase Hour</h4>
                                        <div class="insight-value">${peakPurchaseHour == 'N/A' ? 'N/A' : `${peakPurchaseHour}:00`}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                fadeTransition(summaryContainer, () => {
                    summaryContainer.innerHTML = summaryHTML;
                    
                    // Initialize any charts in the summary
                    if (data.purchase_distribution) {
                        createPurchaseDistributionChart(data.purchase_distribution);
                    }
                });
                
                // Update user analytics plots
                if (data.plots) {
                    // Load each plot with a sequential delay
                    plotIds.forEach((id, index) => {
                        if (data.plots[id]) {
                            setTimeout(() => {
                                const img = document.getElementById(id);
                                if (img) {
                                    const newImg = new Image();
                                    newImg.onload = () => {
                                        img.classList.remove('loading');
                                        fadeTransition(img, () => {
                                            img.src = `data:image/png;base64,${data.plots[id]}`;
                                            img.alt = id.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                        });
                                    };
                                    newImg.src = `data:image/png;base64,${data.plots[id]}`;
                                }
                            }, index * 200);
                        }
                    });
                }
                
                // Re-enable button after all content is loaded
                setTimeout(() => {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-users"></i> Refresh Analytics';
                    showToast('success', 'User Analytics', 'User analytics data loaded', 3000);
                    chartsInitialized = true;
                }, plotIds.length * 200 + 500);
            })
            .catch(error => {
                console.error('Error loading user analytics:', error);
                showToast('error', 'Error', 'Failed to load user analytics: ' + error.message, 5000);
                
                const summaryContainer = document.getElementById('user-analytics-summary');
                summaryContainer.innerHTML = `
                    <div class="error-container" style="text-align:center; padding:30px; color:var(--danger-color);">
                        <i class="fas fa-exclamation-circle" style="font-size:32px; margin-bottom:15px;"></i>
                        <p>Failed to load user analytics data</p>
                        <p>${error.message}</p>
                    </div>
                `;
                
                // Show error state for plots
                plotIds.forEach(id => {
                    const img = document.getElementById(id);
                    if (img) {
                        img.classList.remove('loading');
                        img.src = 'data:image/svg+xml;charset=utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20100%20100%22%3E%3Crect%20width%3D%22100%22%20height%3D%22100%22%20fill%3D%22%23f3f4f6%22%2F%3E%3Ctext%20x%3D%2250%22%20y%3D%2250%22%20font-family%3D%22sans-serif%22%20font-size%3D%2214%22%20text-anchor%3D%22middle%22%20dominant-baseline%3D%22middle%22%20fill%3D%22%236b7280%22%3EFailed%20to%20load%3C%2Ftext%3E%3C%2Fsvg%3E';
                        img.alt = 'Failed to load';
                    }
                });
                
                // Re-enable button
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-users"></i> Try Again';
            });
    }
    
    // Create interactive purchase distribution chart
    function createPurchaseDistributionChart(purchaseData) {
        const chartContainer = document.getElementById('purchase-distribution-chart');
        if (!chartContainer || !purchaseData.purchases_per_user) return;
        
        const ctx = document.createElement('canvas');
        ctx.height = 250;
        chartContainer.appendChild(ctx);
        
        const labels = Object.keys(purchaseData.purchases_per_user);
        const values = Object.values(purchaseData.purchases_per_user);
        
        // Create a simple bar chart with canvas
        const barWidth = Math.max(30, ctx.width / (labels.length + 2));
        const maxValue = Math.max(...values);
        const chartHeight = 200;
        
        ctx.width = barWidth * (labels.length + 2);
        const context = ctx.getContext('2d');
        
        // Draw chart background
        context.fillStyle = '#f9fafb';
        context.fillRect(0, 0, ctx.width, chartHeight + 50);
        
        // Draw title
        context.fillStyle = '#1f2937';
        context.font = 'bold 16px Inter, sans-serif';
        context.textAlign = 'center';
        context.fillText('Purchases per User Distribution', ctx.width / 2, 25);
        
        // Draw bars
        labels.forEach((label, i) => {
            const value = values[i];
            const barHeight = (value / maxValue) * chartHeight;
            const x = barWidth * (i + 1);
            const y = chartHeight - barHeight + 40;
            
            // Draw bar
            context.fillStyle = '#3b82f6';
            context.fillRect(x, y, barWidth * 0.8, barHeight);
            
            // Draw label
            context.fillStyle = '#1f2937';
            context.font = '12px Inter, sans-serif';
            context.textAlign = 'center';
            context.fillText(label, x + barWidth * 0.4, chartHeight + 55);
            
            // Draw value
            context.fillStyle = '#1f2937';
            context.textAlign = 'center';
            context.fillText(value, x + barWidth * 0.4, y - 5);
        });
    }
    
    // Get random session data for prediction
    function getRandomSession() {
        const button = elements.randomSessionBtn;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        
        fetch('/get_random_session')
            .then(handleResponse)
            .then(data => {
                // Fill form with random session data with animation
                fadeTransitionValue('session_events', data.session_events);
                fadeTransitionValue('avg_time_on_page', data.avg_time_on_page);
                fadeTransitionValue('total_time_on_site', data.total_time_on_site);
                fadeTransitionValue('cart_adds', data.cart_adds);
                fadeTransitionValue('cart_to_event_ratio', data.cart_to_event_ratio);
                
                // Set select values if they exist
                if (data.device) {
                    setSelectValue('device', data.device);
                }
                
                if (data.browser) {
                    setSelectValue('browser', data.browser);
                }
                
                // Re-enable button
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-random"></i> Random Session';
                
                // Show toast
                showToast('info', 'Random Session', 'Loaded a random session from the test data', 3000);
            })
            .catch(error => {
                console.error('Error getting random session:', error);
                showToast('error', 'Error', 'Failed to get random session data', 5000);
                
                // Re-enable button
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-random"></i> Random Session';
            });
    }
    
    // Make prediction based on form data
    function makePrediction() {
        const button = elements.predictBtn;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
        
        // Gather form data
        const sessionData = {
            session_events: parseFloat(document.getElementById('session_events').value),
            avg_time_on_page: parseFloat(document.getElementById('avg_time_on_page').value),
            total_time_on_site: parseFloat(document.getElementById('total_time_on_site').value),
            cart_adds: parseFloat(document.getElementById('cart_adds').value),
            cart_to_event_ratio: parseFloat(document.getElementById('cart_to_event_ratio').value),
            device: document.getElementById('device').value,
            browser: document.getElementById('browser').value
        };
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(sessionData)
        })
        .then(handleResponse)
        .then(data => {
            // Show prediction result
            elements.predictionResult.classList.remove('hidden');
            fadeIn(elements.predictionResult);
            
            // Parse probability and animate meter
            const probabilityPercent = parseFloat(data.probability) * 100;
            animateMeter('probability-meter', probabilityPercent);
            
            // Update probability text with counter animation
            animateCounter('probability-text', 0, probabilityPercent, 1500, value => {
                return value.toFixed(1) + '%';
            });
            
            // Set meter color based on probability
            const meterFill = document.getElementById('probability-meter');
            if (probabilityPercent < 30) {
                meterFill.className = 'meter-fill low';
            } else if (probabilityPercent < 70) {
                meterFill.className = 'meter-fill medium';
            } else {
                meterFill.className = 'meter-fill high';
            }
            
            // Update recommendation with typing effect
            typeWriterEffect(document.getElementById('recommendation'), data.recommendation, 30);
            
            // Update contributing factors if available
            if (data.factors) {
                const factorsContainer = document.getElementById('factors-container');
                factorsContainer.innerHTML = '';
                
                data.factors.forEach((factor, index) => {
                    const factorElement = document.createElement('div');
                    factorElement.className = 'factor';
                    factorElement.style.opacity = '0';
                    factorElement.style.transform = 'translateX(-20px)';
                    
                    factorElement.innerHTML = `
                        <span class="factor-name">${factor.name}</span>
                        <div class="factor-impact-container">
                            <div class="factor-impact" style="width: 0%; 
                                 background-color: ${factor.impact > 0 ? 'var(--success-color)' : 'var(--danger-color)'}"></div>
                        </div>
                        <span class="factor-value">${factor.value}</span>
                    `;
                    factorsContainer.appendChild(factorElement);
                    
                    // Animate factor appearance and impact bar
                    setTimeout(() => {
                        factorElement.style.opacity = '1';
                        factorElement.style.transform = 'translateX(0)';
                        factorElement.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        
                        setTimeout(() => {
                            const impactBar = factorElement.querySelector('.factor-impact');
                            impactBar.style.width = Math.abs(factor.impact) * 100 + '%';
                            impactBar.style.transition = 'width 1s ease';
                        }, 300);
                    }, 200 * index);
                });
                
                document.getElementById('factor-analysis').style.display = 'block';
            } else {
                document.getElementById('factor-analysis').style.display = 'none';
            }
            
            // Show toast notification
            const sentimentClass = probabilityPercent > 70 ? 'success' : (probabilityPercent > 30 ? 'warning' : 'info');
            const sentimentText = probabilityPercent > 70 ? 'High' : (probabilityPercent > 30 ? 'Medium' : 'Low');
            showToast(sentimentClass, 'Prediction', `${sentimentText} purchase likelihood: ${data.probability_percent}`, 5000);
            
            // Re-enable button
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-search"></i> Predict';
        })
        .catch(error => {
            console.error('Error making prediction:', error);
            showToast('error', 'Error', 'Failed to make prediction: ' + error.message, 5000);
            
            // Re-enable button
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-search"></i> Predict';
        });
    }
    
    // Show simulation panel
    function showSimulationPanel() {
        elements.simulationPanel.classList.remove('hidden');
        
        // Scroll to the simulation panel with smooth animation
        elements.simulationPanel.scrollIntoView({ behavior: 'smooth' });
        
        // Fade in the panel
        fadeIn(elements.simulationPanel);
    }
    
    // Run session simulation
    function runSessionSimulation() {
        const button = document.getElementById('run-simulation');
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Simulating...';
        
        const numEvents = parseInt(document.getElementById('sim_events').value);
        const includePurchase = document.getElementById('include_purchase').checked;
        
        fetch('/simulate_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_events: numEvents,
                include_purchase: includePurchase
            })
        })
        .then(handleResponse)
        .then(data => {
            // Show simulation result
            document.getElementById('simulation-result').classList.remove('hidden');
            fadeIn(document.getElementById('simulation-result'));
            
            // Create timeline with animation
            createEventTimeline(data.events);
            
            // Update session metrics with animation
            updateSessionMetrics(data.session_metrics);
            
            // Update prediction meter
            if (data.prediction) {
                const probabilityPercent = parseFloat(data.prediction.probability) * 100;
                animateMeter('sim-probability-meter', probabilityPercent);
                
                // Update probability text with counter animation
                animateCounter('sim-probability-text', 0, probabilityPercent, 1500, value => {
                    return value.toFixed(1) + '%';
                });
                
                // Set meter color
                const meterFill = document.getElementById('sim-probability-meter');
                if (probabilityPercent < 30) {
                    meterFill.className = 'meter-fill low';
                } else if (probabilityPercent < 70) {
                    meterFill.className = 'meter-fill medium';
                } else {
                    meterFill.className = 'meter-fill high';
                }
                
                // Show accuracy indicator 
                const accuracyIndicator = document.getElementById('accuracy-indicator');
                if (data.has_purchase) {
                    if (probabilityPercent >= 50) {
                        accuracyIndicator.textContent = '✓ Correct Prediction (True Positive)';
                        accuracyIndicator.className = 'accuracy-indicator correct';
                    } else {
                        accuracyIndicator.textContent = '✗ Incorrect Prediction (False Negative)';
                        accuracyIndicator.className = 'accuracy-indicator incorrect';
                    }
                } else {
                    if (probabilityPercent < 50) {
                        accuracyIndicator.textContent = '✓ Correct Prediction (True Negative)';
                        accuracyIndicator.className = 'accuracy-indicator correct';
                    } else {
                        accuracyIndicator.textContent = '✗ Incorrect Prediction (False Positive)';
                        accuracyIndicator.className = 'accuracy-indicator incorrect';
                    }
                }
            }
            
            // Show toast notification
            showToast('info', 'Simulation Complete', 'User session simulation completed', 3000);
            
            // Re-enable button
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play"></i> Run Simulation';
        })
        .catch(error => {
            console.error('Error running simulation:', error);
            showToast('error', 'Error', 'Failed to run session simulation', 5000);
            
            // Re-enable button
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play"></i> Run Simulation';
        });
    }
    
    // Create animated event timeline
    function createEventTimeline(events) {
        const timeline = document.getElementById('session-timeline');
        timeline.innerHTML = '';
        
        let timelineHTML = '<div class="timeline">';
        
        // Add events one by one with animation
        events.forEach((event, index) => {
            setTimeout(() => {
                const isLast = index === events.length - 1;
                
                let eventIconClass = 'fa-eye';
                let eventClass = 'view-event';
                
                if (event.event_type === 'cart') {
                    eventIconClass = 'fa-cart-plus';
                    eventClass = 'cart-event';
                } else if (event.event_type === 'purchase') {
                    eventIconClass = 'fa-credit-card';
                    eventClass = 'purchase-event';
                }
                
                const eventElement = document.createElement('div');
                eventElement.className = `timeline-item ${eventClass}`;
                eventElement.style.opacity = '0';
                eventElement.style.transform = 'translateY(20px)';
                
                eventElement.innerHTML = `
                    <div class="timeline-icon">
                        <i class="fas ${eventIconClass}"></i>
                    </div>
                    <div class="timeline-content">
                        <h4>Event ${event.event_number}: ${event.event_type}</h4>
                        <p><strong>Time:</strong> ${event.timestamp}</p>
                        <p><strong>Product ID:</strong> ${event.product_id}</p>
                        <p><strong>Time on page:</strong> ${event.time_on_page}s</p>
                    </div>
                    ${!isLast ? '<div class="timeline-connector"></div>' : ''}
                `;
                
                timeline.appendChild(eventElement);
                
                // Animate appearance
                setTimeout(() => {
                    eventElement.style.opacity = '1';
                    eventElement.style.transform = 'translateY(0)';
                    eventElement.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                }, 50);
            }, index * 400); // Stagger appearance of events
        });
    }
    
    // Update session metrics with animation
    function updateSessionMetrics(metrics) {
        const metricsContent = document.getElementById('session-metrics-content');
        
        // Create metrics display
        const metricsHTML = `
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-value" id="events-metric">0</span>
                    <span class="metric-label">Events</span>
                </div>
                <div class="metric">
                    <span class="metric-value" id="cart-adds-metric">0</span>
                    <span class="metric-label">Cart Adds</span>
                </div>
                <div class="metric">
                    <span class="metric-value" id="avg-time-metric">0s</span>
                    <span class="metric-label">Avg Time/Page</span>
                </div>
                <div class="metric">
                    <span class="metric-value" id="total-time-metric">0s</span>
                    <span class="metric-label">Total Time</span>
                </div>
            </div>
            <div class="session-details">
                <p><strong>Device:</strong> ${metrics.device}</p>
                <p><strong>Browser:</strong> ${metrics.browser}</p>
                <p><strong>User ID:</strong> ${metrics.user_id || 'Unknown'}</p>
                <p><strong>Session ID:</strong> ${metrics.session_id || 'Unknown'}</p>
            </div>
        `;
        
        // Update content
        fadeTransition(metricsContent, () => {
            metricsContent.innerHTML = metricsHTML;
            
            // Animate metric counters
            animateCounter('events-metric', 0, metrics.session_events, 1000);
            animateCounter('cart-adds-metric', 0, metrics.cart_adds, 1000);
            animateCounter('avg-time-metric', 0, Math.round(metrics.avg_time_on_page), 1000, value => value + 's');
            animateCounter('total-time-metric', 0, Math.round(metrics.total_time_on_site), 1500, value => value + 's');
        });
    }
    
    // Toggle dataset info panel visibility
    function toggleDatasetInfo() {
        if (elements.datasetInfoPanel.classList.contains('hidden')) {
            elements.showDatasetInfoBtn.disabled = true;
            elements.showDatasetInfoBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            
            fetch('/get_sample_dataset_info')
                .then(handleResponse)
                .then(data => {
                    // Update dataset info
                    updateDatasetInfo(data);
                    
                    // Show the panel with animation
                    elements.datasetInfoPanel.classList.remove('hidden');
                    fadeIn(elements.datasetInfoPanel);
                    
                    // Update button
                    elements.showDatasetInfoBtn.disabled = false;
                    elements.showDatasetInfoBtn.innerHTML = '<i class="fas fa-info-circle"></i> Hide Dataset Info';
                })
                .catch(error => {
                    console.error('Error getting dataset info:', error);
                    showToast('error', 'Error', 'Failed to load dataset information', 5000);
                    
                    elements.showDatasetInfoBtn.disabled = false;
                    elements.showDatasetInfoBtn.innerHTML = '<i class="fas fa-info-circle"></i> Show Dataset Info';
                });
        } else {
            // Hide the panel with animation
            fadeOut(elements.datasetInfoPanel, () => {
                elements.datasetInfoPanel.classList.add('hidden');
            });
            
            // Update button
            elements.showDatasetInfoBtn.innerHTML = '<i class="fas fa-info-circle"></i> Show Dataset Info';
        }
    }
    
    // Update dataset info display
    function updateDatasetInfo(data) {
        const infoContent = document.getElementById('dataset-info-content');
        const info = data.info;
        
        infoContent.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <h4><i class="fas fa-file-alt"></i> Basic Info</h4>
                    <p><strong>File:</strong> ${info.filename}</p>
                    <p><strong>Size:</strong> ${info.file_size}</p>
                    <p><strong>Rows:</strong> ${numberWithCommas(info.rows)}</p>
                    <p><strong>Columns:</strong> ${info.columns}</p>
                </div>
                <div class="info-item">
                    <h4><i class="fas fa-database"></i> Data Elements</h4>
                    <p><strong>Users:</strong> ${numberWithCommas(info.unique_users)}</p>
                    <p><strong>Products:</strong> ${numberWithCommas(info.unique_products)}</p>
                    <p><strong>Sessions:</strong> ${numberWithCommas(info.unique_sessions)}</p>
                    <p><strong>Event Types:</strong> ${info.event_types.join(', ')}</p>
                </div>
                <div class="info-item">
                    <h4><i class="far fa-calendar-alt"></i> Date Range</h4>
                    <p><strong>From:</strong> ${formatDateShort(info.date_range[0])}</p>
                    <p><strong>To:</strong> ${formatDateShort(info.date_range[1])}</p>
                </div>
            </div>
        `;
        
        // Update dataset preview table
        updateDatasetPreview(data.preview);
    }
    
    // Update dataset preview table
    function updateDatasetPreview(preview) {
        const previewTable = document.getElementById('dataset-preview');
        
        if (preview && preview.length > 0) {
            let tableHTML = '<thead><tr>';
            
            // Add headers
            Object.keys(preview[0]).forEach(key => {
                tableHTML += `<th>${key}</th>`;
            });
            
            tableHTML += '</tr></thead><tbody>';
            
            // Add rows
            preview.forEach(row => {
                tableHTML += '<tr>';
                Object.values(row).forEach(value => {
                    tableHTML += `<td>${value}</td>`;
                });
                tableHTML += '</tr>';
            });
            
            tableHTML += '</tbody>';
            previewTable.innerHTML = tableHTML;
        } else {
            previewTable.innerHTML = '<tr><td colspan="5">No preview available</td></tr>';
        }
    }
    
    // Download sample dataset
    function downloadSampleDataset() {
        window.location.href = '/download_sample_dataset';
        showToast('info', 'Downloading', 'Sample dataset download started', 3000);
    }
    
    // =========================================
    // UI Utility Functions
    // =========================================
    
    // Animate progress bar to a specific percentage
    function animateProgressTo(targetPercent) {
        const current = parseInt(elements.progressFill.style.width) || 0;
        const target = parseInt(targetPercent);
        
        if (current === target) return;
        
        const step = 1;
        const interval = 20;
        
        const animation = setInterval(() => {
            const current = parseInt(elements.progressFill.style.width) || 0;
            
            if (current < target) {
                const newValue = Math.min(current + step, target);
                elements.progressFill.style.width = `${newValue}%`;
                elements.progressText.textContent = `${newValue}%`;
            } else {
                clearInterval(animation);
            }
        }, interval);
    }
    
    // Animate meter fill to a specific percentage
    function animateMeter(meterId, targetPercent) {
        const meterFill = document.getElementById(meterId);
        if (!meterFill) return;
        
        meterFill.style.width = '0%';
        
        // Use setTimeout to allow CSS transition to work
        setTimeout(() => {
            meterFill.style.width = `${targetPercent}%`;
        }, 50);
    }
    
    // Animate counter from start to end value
    function animateCounter(elementId, startValue, endValue, duration = 1000, formatter = null) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const start = startValue;
        const end = endValue;
        const range = end - start;
        const increment = range / (duration / 16); // 60fps
        const startTime = performance.now();
        
        const updateCounter = (timestamp) => {
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3); // Cubic ease-out
            const currentValue = start + range * easeProgress;
            
            if (formatter) {
                element.textContent = formatter(Math.round(currentValue));
            } else {
                element.textContent = Math.round(currentValue);
            }
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        };
        
        requestAnimationFrame(updateCounter);
    }
    
    // Show toast notification
    function showToast(type, title, message, duration = 5000) {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toast.innerHTML = `
            <div class="toast-icon"><i class="${icons[type]}"></i></div>
            <div class="toast-body">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">&times;</button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
            toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        }, 10);
        
        // Setup close button
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            removeToast(toast);
        });
        
        // Auto remove after duration
        setTimeout(() => {
            removeToast(toast);
        }, duration);
    }
    
    // Remove toast notification with animation
    function removeToast(toast) {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
    
    // Fade in element
    function fadeIn(element, callback = null) {
        if (!element) return;
        
        element.style.opacity = '0';
        element.style.display = 'block';
        
        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transition = 'opacity 0.5s ease';
            
            if (callback) {
                setTimeout(callback, 500);
            }
        }, 10);
    }
    
    // Fade out element
    function fadeOut(element, callback = null) {
        if (!element) return;
        
        element.style.opacity = '0';
        element.style.transition = 'opacity 0.5s ease';
        
        setTimeout(() => {
            if (element.style.display !== 'none') {
                element.style.display = 'none';
            }
            
            if (callback) {
                callback();
            }
        }, 500);
    }
    
    // Fade transition effect
    function fadeTransition(element, updateFn) {
        if (!element) return;
        
        element.style.opacity = '0';
        element.style.transition = 'opacity 0.3s ease';
        
        setTimeout(() => {
            updateFn();
            
            setTimeout(() => {
                element.style.opacity = '1';
            }, 50);
        }, 300);
    }
    
    // Fade transition for input values
    function fadeTransitionValue(elementId, newValue) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.style.opacity = '0.3';
        element.style.transition = 'opacity 0.3s ease';
        
        setTimeout(() => {
            element.value = newValue;
            element.style.opacity = '1';
        }, 300);
    }
    
    // Set select element value with fade effect
    function setSelectValue(selectId, value) {
        const selectElement = document.getElementById(selectId);
        if (!selectElement) return;
        
        selectElement.style.opacity = '0.3';
        selectElement.style.transition = 'opacity 0.3s ease';
        
        // Find matching option (case insensitive)
        for (let i = 0; i < selectElement.options.length; i++) {
            if (selectElement.options[i].value.toLowerCase() === value.toLowerCase()) {
                setTimeout(() => {
                    selectElement.selectedIndex = i;
                    selectElement.style.opacity = '1';
                }, 300);
                return;
            }
        }
        
        // If no match found, reset opacity
        setTimeout(() => {
            selectElement.style.opacity = '1';
        }, 300);
    }
    
    // Show image in modal
    function showImageModal(src) {
        // Create modal if it doesn't exist
        let modal = document.getElementById('image-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'image-modal';
            modal.className = 'modal';
            
            modal.innerHTML = `
                <span class="modal-close">&times;</span>
                <img class="modal-content" id="modal-image">
            `;
            
            document.body.appendChild(modal);
            
            // Add close button event
            modal.querySelector('.modal-close').addEventListener('click', () => {
                modal.style.display = 'none';
            });
            
            // Close when clicking outside the image
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }
        
        // Show modal with image
        const modalImg = modal.querySelector('#modal-image');
        modalImg.src = src;
        
        modal.style.display = 'flex';
    }
    
    // Type writer effect for status messages
    function typeWriterEffect(element, text, speed = 50) {
        if (!element) return;
        
        let i = 0;
        element.innerHTML = '';
        
        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }
    
    // Show loading state in container
    function showLoading(containerId, rows = 3) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        let loadingHTML = '';
        for (let i = 0; i < rows; i++) {
            loadingHTML += `
                <tr class="loading-row">
                    <td colspan="10">
                        <div class="loading-content"></div>
                    </td>
                </tr>
            `;
        }
        
        container.innerHTML = loadingHTML;
    }
    
       // Tab navigation
    function openTab(evt, tabName) {
        // Get tab id either from data attribute or from event
        const tabId = typeof tabName === 'string' ? tabName : evt.currentTarget.getAttribute('data-tab');
        
        // Hide all tab content
        const tabContent = document.getElementsByClassName('tab-content');
        for (let i = 0; i < tabContent.length; i++) {
            fadeOut(tabContent[i]);
        }
        
        // Remove active class from tab buttons
        const tabButtons = document.getElementsByClassName('tab-btn');
        for (let i = 0; i < tabButtons.length; i++) {
            tabButtons[i].classList.remove('active');
        }
        
        // Show the selected tab and add active class
        const selectedTab = document.getElementById(tabId);
        if (selectedTab) {
            setTimeout(() => {
                selectedTab.classList.add('active');
                fadeIn(selectedTab);
            }, 300);
        }
        
        // Add active class to clicked button
        if (evt && evt.currentTarget) {
            evt.currentTarget.classList.add('active');
        }
    }
    
    // Refresh charts when window resizes
    function refreshChartsOnResize() {
        if (global_data && global_data.user_analytics && global_data.user_analytics.purchase_distribution) {
            createPurchaseDistributionChart(global_data.user_analytics.purchase_distribution);
        }
    }
    
    // =========================================
    // Helper Functions
    // =========================================
    
    // Handle API response
    function handleResponse(response) {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    }
    
    // Handle errors
    function handleError(error) {
        console.error('Error:', error);
        showToast('error', 'Error', error.message || 'An error occurred', 5000);
    }
    
    // Format date for display
    function formatDate(dateString) {
        if (!dateString) return 'N/A';
        
        const date = new Date(dateString);
        if (isNaN(date)) return dateString;
        
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric', 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit',
            hour12: false
        };
        
        return date.toLocaleDateString('en-US', options);
    }
    
    // Format date short version
    function formatDateShort(dateString) {
        if (!dateString) return 'N/A';
        
        const date = new Date(dateString);
        if (isNaN(date)) return dateString;
        
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric'
        };
        
        return date.toLocaleDateString('en-US', options);
    }
    
    // Update all timestamps on the page
    function updateTimestamps() {
        const now = new Date();
        const formattedDate = formatDate(now);
        
        // Update all timestamp elements
        document.getElementById('current-timestamp').textContent = formattedDate;
        document.getElementById('footer-timestamp').textContent = formattedDate;
        document.getElementById('current-year').textContent = now.getFullYear();
    }
    
    // Format number with commas
    function numberWithCommas(x) {
        return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
    
    // Debounce function to limit function calls
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Get current date and time from server or OS
    function getCurrentDateFromServer() {
        fetch('/api/health')
            .then(handleResponse)
            .then(data => {
                if (data.timestamp) {
                    const serverTime = formatDate(data.timestamp);
                    document.getElementById('current-timestamp').textContent = serverTime;
                    document.getElementById('session-start-time').textContent = serverTime;
                    document.getElementById('footer-timestamp').textContent = serverTime;
                }
            })
            .catch(error => {
                console.warn('Could not get server time:', error);
                // Fall back to client time which is already set
            });
    }
    
    // Initialize the application once page is loaded
    document.addEventListener('DOMContentLoaded', () => {
        // Set current user info if provided
        // Note: In real deployment, this should come from server
        const userLogin = 'admin';
        const userElements = document.querySelectorAll('.user-login');
        userElements.forEach(el => {
            el.textContent = userLogin;
        });
        
        // Try to get current server time
        getCurrentDateFromServer();
        
        // Check if there's a hash in URL for direct tab access
        const hash = window.location.hash.substring(1);
        if (hash && document.getElementById(hash)) {
            // Find the tab button
            const tabButton = document.querySelector(`.tab-btn[data-tab="${hash}"]`);
            if (tabButton) {
                tabButton.click();
            } else {
                // Manually open the tab if button not found
                openTab(null, hash);
            }
        }
        
        // Initialize any pre-loaded data
        initializePreloadedData();
    });
    
    // Initialize any data that might be preloaded
    function initializePreloadedData() {
        // Check if there's any preloaded data in the page
        const preloadedDataElement = document.getElementById('preloaded-data');
        if (preloadedDataElement) {
            try {
                const preloadedData = JSON.parse(preloadedDataElement.textContent);
                if (preloadedData.results) {
                    modelData = preloadedData.results;
                }
                if (preloadedData.timestamp) {
                    document.getElementById('current-timestamp').textContent = formatDate(preloadedData.timestamp);
                    document.getElementById('session-start-time').textContent = formatDate(preloadedData.timestamp);
                    document.getElementById('footer-timestamp').textContent = formatDate(preloadedData.timestamp);
                }
                if (preloadedData.userLogin) {
                    const userElements = document.querySelectorAll('.user-login');
                    userElements.forEach(el => {
                        el.textContent = preloadedData.userLogin;
                    });
                }
            } catch (error) {
                console.warn('Error parsing preloaded data:', error);
            }
        }
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // ESC key closes modals
        if (e.key === 'Escape') {
            const modal = document.getElementById('image-modal');
            if (modal && modal.style.display === 'flex') {
                modal.style.display = 'none';
            }
        }
        
        // Ctrl+P for prediction
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            const predictBtn = document.getElementById('predict-btn');
            if (predictBtn && !predictBtn.disabled) {
                predictBtn.click();
            }
        }
    });
    
    // Handle automatic timestamp updates for admin user
    function startAutomaticTimestamps() {
        // Update timestamp every second
        setInterval(() => {
            // Only update if user is admin
            if (userLogin === 'admin') {
                const now = new Date();
                const formattedTimestamp = formatDateTime(now);
                
                // Update timestamps in various places
                document.querySelectorAll('.dynamic-timestamp').forEach(el => {
                    el.textContent = formattedTimestamp;
                });
            }
        }, 1000);
    }
    
    // Format date to UTC with specific format
    function formatDateTime(date) {
        return date.toISOString().replace('T', ' ').substring(0, 19);
    }
    
    // Start automatic timestamp updates
    startAutomaticTimestamps();
});