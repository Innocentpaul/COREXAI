// COREXAI Visual Explainers - JavaScript
// Handles UI interactions, API requests, and visualization rendering

document.addEventListener('DOMContentLoaded', function() {
    // Constants
    const API_ENDPOINT = '/explain';
    const MODELS = {
        'intrusion': {
            name: 'Intrusion Detection',
            description: 'AI model trained to detect network intrusions and security anomalies.',
            features: 30
        },
        'fraud': {
            name: 'Financial Fraud Detection',
            description: 'AI model trained to identify fraudulent financial transactions.',
            features: 30
        }
    };

    // Element references
    const modelSelect = document.getElementById('modelSelect');
    const explanationType = document.getElementById('explanationType');
    const numSamples = document.getElementById('numSamples');
    const numSamplesValue = document.getElementById('numSamplesValue');
    const topFeatures = document.getElementById('topFeatures');
    const topFeaturesValue = document.getElementById('topFeaturesValue');
    const useCustomData = document.getElementById('useCustomData');
    const customDataContainer = document.getElementById('customDataContainer');
    const customDataInput = document.getElementById('customDataInput');
    const generateBtn = document.getElementById('generateBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const themeToggle = document.getElementById('themeToggle');
    const helpBtn = document.getElementById('helpBtn');
    const helpModal = document.getElementById('helpModal');
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const imageContainer = document.getElementById('imageContainer');
    const explanationImage = document.getElementById('explanationImage');
    const plotlyContainer = document.getElementById('plotlyContainer');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceMeter = document.getElementById('confidenceMeter');
    const probabilityChart = document.getElementById('probabilityChart');
    const featureSearch = document.getElementById('featureSearch');
    const featureList = document.getElementById('featureList');
    const featureInputsContainer = document.getElementById('featureInputs');

    // State
    let currentExplanation = null;
    let isDarkTheme = localStorage.getItem('corexai-theme') !== 'light';
    
    // Initialize theme
    function initTheme() {
        if (isDarkTheme) {
            document.body.classList.remove('light-theme');
            themeToggle.innerHTML = '<i class="fa-solid fa-moon"></i>';
        } else {
            document.body.classList.add('light-theme');
            themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
        }
    }
    
    // Theme toggle
    themeToggle.addEventListener('click', function() {
        isDarkTheme = !isDarkTheme;
        localStorage.setItem('corexai-theme', isDarkTheme ? 'dark' : 'light');
        initTheme();
    });

    // Update explanation type options based on selected type
    explanationType.addEventListener('change', function() {
        const selectedType = this.value;
        
        // Show/hide num samples based on explanation type
        if (selectedType === 'shap_beeswarm') {
            numSamples.min = 10;
            numSamples.value = Math.max(10, numSamples.value);
            numSamplesValue.textContent = numSamples.value;
        } else {
            numSamples.min = 1;
        }
        
        // Update help text based on selection
        updateHelpText(selectedType);
    });

    function updateHelpText(explType) {
        const helpTexts = {
            'shap_2d': 'Waterfall plot showing how each feature contributes to the prediction',
            'shap_3d': 'Interactive 3D visualization of feature contributions',
            'shap_beeswarm': 'Shows feature impact distribution across multiple samples',
            'shap_force': 'Visualizes how features push the prediction from the base value',
            'lime_2d': 'Local explanation showing feature importance for this specific prediction',
            'lime_3d': 'Interactive 3D visualization of LIME feature impacts',
            'feature_correlation': 'Heatmap showing correlations between top features',
            'feature_comparison': 'Compares feature importance across different methods'
        };
        
        // You could display this help text somewhere in the UI
        console.log(helpTexts[explType] || 'Generate explanations to visualize AI decisions');
    }

    // Create feature input fields based on selected model
    function createFeatureInputFields(modelType) {
        if (!featureInputsContainer) return;
        
        featureInputsContainer.innerHTML = '';
        
        const numFeatures = MODELS[modelType].features;
        const featureNames = [];
        
        for (let i = 0; i < numFeatures; i++) {
            featureNames.push(`${modelType === 'intrusion' ? 'Intrusion' : 'Fraud'}_Feature_${i+1}`);
        }
        
        const title = document.createElement('h3');
        title.textContent = 'Sample Feature Values (first 5 shown)';
        title.style.cssText = 'font-size: 0.9rem; margin: 15px 0 10px 0; color: var(--text-secondary);';
        featureInputsContainer.appendChild(title);
        
        const note = document.createElement('p');
        note.className = 'feature-note';
        note.textContent = `All ${numFeatures} features will be randomly generated for demonstration.`;
        note.style.cssText = 'font-size: 0.8rem; margin-bottom: 10px; font-style: italic; color: var(--text-secondary);';
        featureInputsContainer.appendChild(note);
        
        const displayedFeatures = Math.min(5, featureNames.length);
        for (let i = 0; i < displayedFeatures; i++) {
            const inputGroup = document.createElement('div');
            inputGroup.className = 'feature-input-group';
            inputGroup.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 8px;';
            
            const label = document.createElement('label');
            label.textContent = featureNames[i];
            label.setAttribute('for', `feature_${i}`);
            label.style.cssText = 'flex: 1; font-size: 0.85rem;';
            
            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.id = `feature_${i}`;
            input.className = 'feature-input styled-input';
            input.value = (Math.random()).toFixed(4);
            input.style.cssText = 'width: 100px; padding: 5px; border-radius: 4px; background-color: var(--input-bg); border: 1px solid var(--border-color); color: var(--text-color);';
            
            inputGroup.appendChild(label);
            inputGroup.appendChild(input);
            featureInputsContainer.appendChild(inputGroup);
        }
    }
    
    // Update feature inputs when model changes
    modelSelect.addEventListener('change', function() {
        createFeatureInputFields(this.value);
    });

    // Range input handlers
    numSamples.addEventListener('input', function() {
        numSamplesValue.textContent = this.value;
    });
    
    topFeatures.addEventListener('input', function() {
        topFeaturesValue.textContent = this.value;
    });
    
    // Custom data toggle
    useCustomData.addEventListener('change', function() {
        customDataContainer.style.display = this.checked ? 'block' : 'none';
        featureInputsContainer.style.display = this.checked ? 'none' : 'block';
    });
    
    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            this.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });
    
    // Help modal
    helpBtn.addEventListener('click', function() {
        helpModal.classList.add('active');
    });
    
    document.querySelector('.close-button').addEventListener('click', function() {
        helpModal.classList.remove('active');
    });
    
    window.addEventListener('click', function(event) {
        if (event.target === helpModal) {
            helpModal.classList.remove('active');
        }
    });
    
    // Feature search
    featureSearch.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const features = featureList.querySelectorAll('.feature-item');
        
        features.forEach(feature => {
            const featureName = feature.querySelector('.feature-name').textContent.toLowerCase();
            feature.style.display = featureName.includes(searchTerm) ? 'flex' : 'none';
        });
    });
    
    // Generate random feature values
    function generateRandomFeatureValues(modelType) {
        const numFeatures = MODELS[modelType].features;
        const values = [];
        
        for (let i = 0; i < numFeatures; i++) {
            values.push(parseFloat((Math.random()).toFixed(4)));
        }
        
        return values;
    }
    
    // Generate explanation
    generateBtn.addEventListener('click', async function() {
        try {
            // Show loading overlay
            loadingOverlay.classList.add('active');
            
            // Get selected options
            const model = modelSelect.value;
            const explType = explanationType.value;
            const samples = parseInt(numSamples.value);
            const features = parseInt(topFeatures.value);
            
            // Prepare request data
            const requestData = {
                model_type: model,
                explanation_type: explType,
                num_samples: samples,
                top_features: features
            };
            
            // Get or generate input data
            let inputData = [];
            
            if (useCustomData.checked && customDataInput.value.trim()) {
                try {
                    inputData = JSON.parse(customDataInput.value.trim());
                    if (!Array.isArray(inputData)) {
                        throw new Error('Custom data must be an array');
                    }
                    
                    const expectedCount = MODELS[model].features;
                    if (inputData.length !== expectedCount) {
                        throw new Error(`Expected ${expectedCount} values, but got ${inputData.length}`);
                    }
                } catch (error) {
                    showError(`Invalid custom data: ${error.message}`);
                    return;
                }
            } else {
                inputData = generateRandomFeatureValues(model);
            }
            
            requestData.input_data = inputData;
            
            // Call API
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `API error (${response.status})`);
            }
            
            const data = await response.json();
            currentExplanation = data;
            
            // Update UI with results
            updateVisualization(data);
            updatePredictionInfo(data);
            updateFeatureList(data, inputData);
            downloadBtn.disabled = false;
            
            // Switch to visualization tab
            document.querySelector('[data-tab="visualization"]').click();
            
        } catch (error) {
            showError(`Error generating explanation: ${error.message}`);
        } finally {
            loadingOverlay.classList.remove('active');
        }
    });
    
    // Download results
    downloadBtn.addEventListener('click', function() {
        if (!currentExplanation) return;
        
        const explType = explanationType.value;
        
        if (currentExplanation.type === 'html') {
            // For HTML content, create a downloadable HTML file
            const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>${explType} Explanation - COREXAI</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>${explType.replace(/_/g, ' ').toUpperCase()} Explanation</h1>
    <div id="plot">${currentExplanation.plot}</div>
</body>
</html>`;
            
            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            downloadFile(url, `${explType}_explanation.html`);
        } else {
            // For images, download the base64 image
            const link = document.createElement('a');
            link.href = currentExplanation.plot;
            link.download = `${explType}_explanation.png`;
            link.click();
        }
    });
    
    // Update visualization
    function updateVisualization(data) {
        if (!data || !data.plot) return;
        
        if (data.type === 'html') {
            // Handle interactive plots
            imageContainer.style.display = 'none';
            plotlyContainer.style.display = 'block';
            
            // For 3D plots, the HTML includes Plotly inline
            plotlyContainer.innerHTML = data.plot;
            
        } else {
            // Handle static images
            imageContainer.style.display = 'block';
            plotlyContainer.style.display = 'none';
            explanationImage.src = data.plot;
            explanationImage.alt = `${explanationType.value} explanation visualization`;
        }
    }
    
    // Update prediction information
    function updatePredictionInfo(data) {
        if (!data || !data.predictions || !data.probabilities) return;
        
        const prediction = data.predictions[0];
        const probabilityArray = data.probabilities[0];
        
        // Determine label
        const labels = modelSelect.value === 'intrusion' 
            ? ['Normal', 'Malicious'] 
            : ['Legitimate', 'Fraudulent'];
        
        const label = labels[prediction];
        
        // Update prediction label
        predictionLabel.textContent = label;
        predictionLabel.className = 'prediction-label';
        predictionLabel.classList.add(prediction === 1 ? 'text-danger' : 'text-success');
        
        // Update confidence meter
        const confidence = probabilityArray[prediction];
        const confidencePercent = Math.round(confidence * 100);
        
        confidenceMeter.style.width = `${confidencePercent}%`;
        confidenceMeter.textContent = `${confidencePercent}%`;
        
        // Update confidence color
        if (confidencePercent > 80) {
            confidenceMeter.style.backgroundColor = 'var(--success-color)';
        } else if (confidencePercent > 60) {
            confidenceMeter.style.backgroundColor = 'var(--accent-color)';
        } else if (confidencePercent > 40) {
            confidenceMeter.style.backgroundColor = 'var(--warning-color)';
        } else {
            confidenceMeter.style.backgroundColor = 'var(--danger-color)';
        }
        
        // Generate probability chart
        generateProbabilityChart(probabilityArray, labels);
    }
    
    // Generate probability chart
    function generateProbabilityChart(probabilities, labels) {
        probabilityChart.innerHTML = '';
        
        const chartHtml = `
            <div class="probability-bars">
                ${probabilities.map((value, index) => `
                    <div class="prob-bar-container">
                        <div class="prob-bar-label">${labels[index]}</div>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar" style="width: ${value * 100}%; 
                                background-color: ${index === 1 ? 'var(--danger-color)' : 'var(--success-color)'}">
                                ${Math.round(value * 100)}%
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        probabilityChart.innerHTML = chartHtml;
        
        // Add styles if not already present
        if (!document.getElementById('prob-chart-styles')) {
            const style = document.createElement('style');
            style.id = 'prob-chart-styles';
            style.textContent = `
                .probability-bars {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    width: 100%;
                }
                .prob-bar-container {
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }
                .prob-bar-label {
                    width: 100px;
                    font-weight: 600;
                    text-align: right;
                    color: var(--text-color);
                }
                .prob-bar-wrapper {
                    flex: 1;
                    background-color: var(--input-bg);
                    height: 30px;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
                }
                .prob-bar {
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 600;
                    font-size: 0.85rem;
                    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
                    min-width: 60px;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Update feature list
    function updateFeatureList(data, featureValues) {
        featureList.innerHTML = '';
        
        if (!data || !data.feature_impacts) {
            featureList.innerHTML = '<p class="empty-state">No feature impact data available</p>';
            return;
        }
        
        const modelType = modelSelect.value;
        const featureNames = [];
        for (let i = 0; i < MODELS[modelType].features; i++) {
            featureNames.push(`${modelType === 'intrusion' ? 'Intrusion' : 'Fraud'}_Feature_${i+1}`);
        }
        
        // Create feature objects with impacts
        const features = featureNames.map((name, index) => ({
            name: name,
            value: featureValues[index],
            impact: data.feature_impacts[index] || 0
        }));
        
        // Sort by absolute impact
        features.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
        
        // Create feature items
        features.slice(0, 20).forEach(feature => {
            const featureItem = document.createElement('div');
            featureItem.className = 'feature-item';
            
            // Determine impact indicator
            let impactIcon = '';
            let impactClass = '';
            
            if (feature.impact > 0.1) {
                impactIcon = '<i class="fa-solid fa-arrow-up"></i>';
                impactClass = 'text-danger';
            } else if (feature.impact < -0.1) {
                impactIcon = '<i class="fa-solid fa-arrow-down"></i>';
                impactClass = 'text-success';
            } else {
                impactIcon = '<i class="fa-solid fa-minus"></i>';
                impactClass = 'text-secondary';
            }
            
            featureItem.innerHTML = `
                <div class="feature-info">
                    <span class="feature-name">${feature.name}</span>
                    <span class="feature-impact ${impactClass}">
                        ${impactIcon} ${Math.abs(feature.impact).toFixed(4)}
                    </span>
                </div>
                <div class="feature-value">${parseFloat(feature.value).toFixed(4)}</div>
            `;
            
            featureList.appendChild(featureItem);
        });
        
        // Add styles if not already present
        if (!document.getElementById('feature-list-styles')) {
            const style = document.createElement('style');
            style.id = 'feature-list-styles';
            style.textContent = `
                .feature-info {
                    display: flex;
                    gap: 15px;
                    align-items: center;
                    flex: 1;
                }
                .feature-impact {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    font-weight: 600;
                }
                .text-danger { color: var(--danger-color); }
                .text-success { color: var(--success-color); }
                .text-secondary { color: var(--text-secondary); }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Helper function to download a file
    function downloadFile(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        if (url.startsWith('blob:')) {
            URL.revokeObjectURL(url);
        }
    }
    
    // Show error message
    function showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'error-alert';
        alertDiv.innerHTML = `
            <div class="error-content">
                <i class="fa-solid fa-circle-exclamation"></i>
                <span>${message}</span>
                <button class="close-alert">&times;</button>
            </div>
        `;
        
        // Add styles if not already present
        if (!document.getElementById('error-alert-styles')) {
            const style = document.createElement('style');
            style.id = 'error-alert-styles';
            style.textContent = `
                .error-alert {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1010;
                    animation: slideIn 0.3s ease-out;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
                .error-content {
                    background-color: var(--danger-color);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    max-width: 400px;
                }
                .close-alert {
                    background: transparent;
                    border: none;
                    color: white;
                    font-size: 1.2rem;
                    cursor: pointer;
                    margin-left: 10px;
                    padding: 0 5px;
                }
                .close-alert:hover {
                    opacity: 0.8;
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        const timeout = setTimeout(() => {
            alertDiv.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                document.body.removeChild(alertDiv);
            }, 300);
        }, 5000);
        
        // Close button handler
        alertDiv.querySelector('.close-alert').addEventListener('click', () => {
            clearTimeout(timeout);
            alertDiv.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                document.body.removeChild(alertDiv);
            }, 300);
        });
    }
    
    // Initialize
    function initialize() {
        // Set initial values
        numSamplesValue.textContent = numSamples.value;
        topFeaturesValue.textContent = topFeatures.value;
        
        // Create initial feature inputs
        createFeatureInputFields(modelSelect.value);
        
        // Initialize theme
        initTheme();
        
        // Load placeholder image
        explanationImage.src = '/static/images/placeholder-chart.png';
    }
    
    // Start initialization
    initialize();
});