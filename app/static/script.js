// Mental Health Monitoring System - Frontend Script

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const submittedData = {};

    // Update value displays as user changes inputs
    const numericInputs = form.querySelectorAll('input[type="number"], input[type="range"]');
    numericInputs.forEach(input => {
        input.addEventListener('change', function() {
            updateValueDisplay(this);
        });
        input.addEventListener('input', function() {
            updateValueDisplay(this);
        });
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(form);
        // Aggregate cognitive and emotional sliders (capture any questionnaire names)
        const cognitiveKeys = ['cognitive_state','cognitive_state_2','concentration','mental_tired'];
        const emotionalKeys = ['emotional_state','emotional_state_primary','emotional_state_secondary','overloaded','emotional_state_2','emotional_state_3'];

        const cognitiveVals = [];
        cognitiveKeys.forEach(k => { if (formData.get(k)) cognitiveVals.push(parseInt(formData.get(k))); });
        const cognitive_state = cognitiveVals.length ? Math.round(cognitiveVals.reduce((a,b)=>a+b,0)/cognitiveVals.length) : parseInt(formData.get('cognitive_state') || 3);

        const emotionalVals = [];
        emotionalKeys.forEach(k => { if (formData.get(k)) emotionalVals.push(parseInt(formData.get(k))); });
        const emotional_state = emotionalVals.length ? Math.round(emotionalVals.reduce((a,b)=>a+b,0)/emotionalVals.length) : parseInt(formData.get('emotional_state') || 3);

        const data = {
            heart_rate: parseFloat(formData.get('heart_rate')),
            hrv: parseFloat(formData.get('hrv')),
            respiration: parseFloat(formData.get('respiration')),
            skin_temp: parseFloat(formData.get('skin_temp')),
            bp_systolic: parseFloat(formData.get('bp_systolic')),
            bp_diastolic: parseFloat(formData.get('bp_diastolic')),
            cognitive_state: cognitive_state,
            emotional_state: emotional_state
        };

        // Store submitted data
        Object.assign(submittedData, data);

        // Show loading indicator
        showLoading(true);
        hideError();

        try {
            // Send prediction request to API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-ID': getOrCreateUserId()
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get prediction');
            }

            const result = await response.json();
            displayResults(result);
            // Don't call updateTrendChart here - renderTrendChart() in displaySimpleResults handles it

        } catch (error) {
            showError(error.message);
            console.error('Error:', error);
        } finally {
            showLoading(false);
        }
    });

    function updateValueDisplay(input) {
        // Try to find .value-display in parent (used in older layout)
        let display = input.parentElement.querySelector('.value-display');
        if (display) {
            display.textContent = parseFloat(input.value).toFixed(1);
            return;
        }
        // Try to find .metric-display-value > .value-display for new metric cards
        const metricCard = input.closest('.metric-card');
        if (metricCard) {
            display = metricCard.querySelector('.metric-display-value .value-display');
            if (display) {
                display.textContent = parseFloat(input.value).toFixed(1);
                return;
            }
        }
    }

    function displayResults(result) {
        // Check for errors
        if (result.error) {
            showError(result.error);
            return;
        }

        const resultsSection = document.getElementById('resultsSection');
        
        // Handle complex API responses first, then legacy simple responses
        if (result.ml_prediction || result.mental_load_index) {
            displayComplexResults(result);
        } else if (result.stress_level) {
            displaySimpleResults(result);
        } else {
            showError('Unexpected response format');
        }
        
        // Show results section
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    function displaySimpleResults(result) {
        // Build dissertation-structured results (no raw ML metrics shown)
        const conditionLabel = document.getElementById('conditionLabel');
        const mliScore = document.getElementById('mliScore');
        const mliInterpret = document.getElementById('mliInterpret');
        const advisoryList = document.getElementById('advisoryList');
        const trendInsight = document.getElementById('trendInsight');
        const warningMessage = document.getElementById('warningMessage');
        const naturalList = document.getElementById('naturalInterventions');
        const otcList = document.getElementById('otcOptions');
        const profList = document.getElementById('professionalServices');

        // Determine condition mapping
        const condition = mapCondition(result);
        if (conditionLabel) conditionLabel.textContent = `Condition: ${condition}`;

        // Compute Mental Load Index (0-100)
        const mli = computeMLI(result);

        // Animate gauge and set value
        setGauge(mli);

        // Interpretation (broad) and Category (granular)
        const interpret = mli <= 30 ? 'Calm' : (mli <= 70 ? 'Moderate' : 'Stressed');
        // Replace 'Stressed' label with a more professional phrasing when high
        const interpText = (interpret === 'Stressed') ? 'Elevated Cognitive Load Detected' : interpret;
        if (mliInterpret) mliInterpret.textContent = interpText;
        const mliCategoryEl = document.getElementById('mliCategory');
        let categoryText = '';
        if (mli <= 30) categoryText = 'Calm';
        else if (mli <= 50) categoryText = 'Moderate–Low';
        else if (mli <= 70) categoryText = 'Moderate–High';
        else categoryText = 'Stressed';
        if (mliCategoryEl) mliCategoryEl.textContent = `Category: ${categoryText}`;

        // Advisory sentences (3-4 lines) based on condition
        const adv = getAdvisoryForCondition(condition);
        if (advisoryList) {
            advisoryList.innerHTML = adv.map(s => `<p class="advice-line">• ${s}</p>`).join('');
        }

        // Early warning
        const warn = getEarlyWarningForCondition(condition);
        if (warningMessage) warningMessage.textContent = warn;

        // Personalized tiered recommendations
        const tiers = getTieredRecommendations(condition);
        if (naturalList) naturalList.innerHTML = tiers.natural.map(i => `<li>${i}</li>`).join('');
        if (otcList) otcList.innerHTML = tiers.otc.map(i => `<li>${i}</li>`).join('');
        if (profList) profList.innerHTML = tiers.professional.map(i => `<li>${i}</li>`).join('');

        // Store history and update trend chart
        appendToHistory({ timestamp: new Date().toISOString(), mli: mli, condition: condition });
        renderTrendChart();

        // Update condition status cards (visual)
        try {
            document.getElementById('conditionLabel').textContent = `Condition: ${condition}`;
            ['status-calm','status-moderate','status-stressed'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });
            if (categoryText === 'Calm') document.getElementById('status-calm').classList.add('active');
            else if (categoryText.includes('Moderate')) document.getElementById('status-moderate').classList.add('active');
            else document.getElementById('status-stressed').classList.add('active');
        } catch(e) { /* ignore */ }

        // Trend insight
        if (trendInsight) trendInsight.textContent = getTrendInsight(condition);
    }

    // Animate and set gauge value (circular SVG)
    function setGauge(value) {
        const circle = document.getElementById('gaugeProgress');
        const pulse = document.getElementById('gaugePulse');
        const valueText = document.getElementById('mliGaugeValue');
        if (!circle || !valueText) return;
        const radius = 44; // same as SVG r
        const circumference = 2 * Math.PI * radius;
        const offset = Math.max(0, circumference - (value / 100) * circumference);
        circle.style.strokeDashoffset = offset;
        // change stroke color using gradient stops (visual effect already styled in CSS)
        valueText.textContent = `${value}`;
        if (pulse) {
            if (value >= 71) pulse.classList.add('pulse'); else pulse.classList.remove('pulse');
        }
    }

    function displayMLIComponents(components) {
        const componentsDiv = document.getElementById('mliComponents');
        
        if (!componentsDiv || !components || typeof components !== 'object') return;

        componentsDiv.innerHTML = '';

        const componentNames = {
            'heart_rate_stress': 'Heart Rate',
            'hrv_stress': 'HRV',
            'respiration_stress': 'Respiration',
            'temperature_stress': 'Temperature',
            'blood_pressure_stress': 'Blood Pressure',
            'cognitive_stress': 'Cognitive Load',
            'emotional_stress': 'Emotional Stress'
        };

        for (const [key, value] of Object.entries(components)) {
            const displayName = componentNames[key] || key;
            const percentage = (value * 100).toFixed(1);
            
            const componentHTML = `
                <div class="component-bar">
                    <div class="component-label">${displayName}</div>
                    <div class="component-bar-bg">
                        <div class="component-bar-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.85em; color: #666;">${percentage}%</div>
                </div>
            `;
            
            componentsDiv.innerHTML += componentHTML;
        }
    }

    // ---------------------- Helper functions for new result layout ----------------------
    function mapCondition(result) {
        const lvl = (result.stress_level || '').toLowerCase();
        if (lvl === 'low' || lvl === 'calm') return 'Calm';
        if (lvl === 'high') return 'Stressed';
        // For moderate-low or moderate-high or ambiguous, map to Moderate
        return 'Moderate';
    }

    function computeMLI(result) {
        if (result.mental_load_index !== undefined && result.mental_load_index !== null) {
            if (typeof result.mental_load_index === 'object' && result.mental_load_index.score !== undefined) {
                let mli = parseFloat(result.mental_load_index.score);
                if (isNaN(mli)) mli = 50;
                return Math.min(100, Math.max(0, Math.round(mli)));
            }
            let mli = parseFloat(result.mental_load_index);
            if (!isNaN(mli)) {
                return Math.min(100, Math.max(0, Math.round(mli)));
            }
        }
        return 50;
    }

    function getCategoryText(mli, result) {
        if (result && result.mental_load_index && result.mental_load_index.category) {
            return result.mental_load_index.category;
        }
        if (mli <= 30) return 'Calm';
        if (mli <= 50) return 'Moderate–Low';
        if (mli <= 70) return 'Moderate–High';
        return 'Stressed';
    }

    function displayComplexResults(result) {
        const conditionLabel = document.getElementById('conditionLabel');
        const mliScore = document.getElementById('mliGaugeValue');
        const mliInterpret = document.getElementById('mliInterpret');
        const mliCategoryEl = document.getElementById('mliCategory');
        const advisoryList = document.getElementById('advisoryList');
        const warningMessage = document.getElementById('warningMessage');
        const naturalInterventions = document.getElementById('naturalInterventions');
        const otcOptions = document.getElementById('otcOptions');
        const professionalServices = document.getElementById('professionalServices');
        const predictedStress = document.getElementById('predictedStress');
        const modelConfidence = document.getElementById('modelConfidence');

        const mli = computeMLI(result);
        const categoryText = getCategoryText(mli, result);
        const condition = result.ml_prediction ? normalizeCondition(result.ml_prediction.stress_level) : normalizeCondition(result.stress_level || result.condition);

        if (conditionLabel) {
            conditionLabel.textContent = `Condition: ${condition}`;
        }

        if (mliScore) {
            mliScore.textContent = `${mli}`;
        }

        setGauge(mli);

        if (mliInterpret) {
            mliInterpret.textContent = result.mental_load_index?.status || categoryText;
        }

        if (mliCategoryEl) {
            mliCategoryEl.textContent = `Category: ${categoryText}`;
        }

        if (advisoryList) {
            const advisoryItems = (result.advisory?.recommendations) || getAdvisoryForCondition(condition);
            if (advisoryItems.length) {
                advisoryList.innerHTML = advisoryItems.map(item => `<p class="advice-line">• ${item}</p>`).join('');
            } else {
                advisoryList.innerHTML = '<p class="advice-line"><em>No advisory information available.</em></p>';
            }
        }

        if (warningMessage) {
            warningMessage.textContent = result.early_warning?.escalation_risk || result.early_warning?.burnout_risk_level || getEarlyWarningForCondition(condition);
        }

        const recData = result.recommendations || result.recommendation;
        const naturalItems = recData.natural_interventions || recData.natural || recData.actions || [];
        const otcItems = recData.otc_options || recData.otc || [];
        const professionalItems = recData.professional_services || recData.professional || [];

        if (naturalInterventions) {
            naturalInterventions.innerHTML = `<ul class="rec-list-items">${naturalItems.length ? naturalItems.map(item => `<li>${item}</li>`).join('') : '<li><em>No natural interventions available.</em></li>'}</ul>`;
        }
        if (otcOptions) {
            otcOptions.innerHTML = `<ul class="rec-list-items">${otcItems.length ? otcItems.map(item => `<li>${item}</li>`).join('') : '<li><em>No OTC options available.</em></li>'}</ul>`;
        }
        if (professionalServices) {
            professionalServices.innerHTML = `<ul class="rec-list-items">${professionalItems.length ? professionalItems.map(item => `<li>${item}</li>`).join('') : '<li><em>No professional services recommended.</em></li>'}</ul>`;
        }

        if (predictedStress) {
            predictedStress.textContent = condition;
        }
        if (modelConfidence && result.ml_prediction?.confidence !== undefined) {
            modelConfidence.textContent = `${(result.ml_prediction.confidence * 100).toFixed(1)}%`;
        }

        if (result.ml_prediction?.probabilities) {
            displayProbabilityChart(result.ml_prediction.probabilities);
        }

        try {
            document.getElementById('status-calm')?.classList.remove('active');
            document.getElementById('status-moderate')?.classList.remove('active');
            document.getElementById('status-stressed')?.classList.remove('active');
            if (categoryText === 'Calm') document.getElementById('status-calm')?.classList.add('active');
            else if (categoryText.includes('Moderate')) document.getElementById('status-moderate')?.classList.add('active');
            else document.getElementById('status-stressed')?.classList.add('active');
        } catch (e) {
            console.error('Status card update error', e);
        }

        appendToHistory({ timestamp: new Date().toISOString(), mli: mli, condition: condition });
        renderTrendChart();
    }

    function getAdvisoryForCondition(condition) {
        if (condition === 'Calm') return ['Maintain current routine', 'Light physical activity recommended', 'Continue balanced sleep cycle'];
        if (condition === 'Moderate') return ['Take short recovery breaks', 'Reduce multitasking', 'Practice breathing exercises', 'Prioritize rest tonight'];
        return ['Immediate recovery is advised', 'Avoid demanding cognitive tasks', 'Engage in guided relaxation', 'Reduce environmental stress exposure'];
    }

    function normalizeCondition(category) {
        if (!category) return 'Moderate';
        const normalized = category.toString().trim().toLowerCase();
        if (normalized === 'low' || normalized === 'calm') return 'Calm';
        if (normalized === 'high' || normalized === 'stressed') return 'Stressed';
        return 'Moderate';
    }

    function getEarlyWarningForCondition(condition) {
        if (condition === 'Calm') return 'No immediate risk predicted.';
        if (condition === 'Moderate') return 'If current pattern continues, high stress state may develop within 48–72 hours.';
        return 'High burnout risk if current pattern persists. Immediate intervention recommended.';
    }

    function getTrendInsight(condition) {
        if (condition === 'Calm') return 'Stress levels have remained stable over the past week.';
        if (condition === 'Moderate') return 'Gradual increase in mental load observed in recent days.';
        return 'Consistent upward stress pattern detected. Recovery appears limited.';
    }

    function getTieredRecommendations(condition) {
        if (condition === 'Calm') return {
            natural: ['Light yoga or stretching', '10-minute mindfulness', 'Continue balanced sleep cycle', 'Short daily walk'],
            otc: ['Not required'],
            professional: ['Not required']
        };
        if (condition === 'Moderate') return {
            natural: ['Ashwagandha support', 'Brahmi for cognitive clarity', '15-minute breathing practice', 'Limit caffeine after noon'],
            otc: ['Magnesium supplementation', 'Herbal calming formulation', 'Consider melatonin for sleep support'],
            professional: ['Consider consultation if persists for several days']
        };
        return {
            natural: ['Immediate relaxation protocol', 'Digital detox period', 'Short guided breathing protocol'],
            otc: ['Short-term sleep support (if needed)', 'Short course sleep aids as advised'],
            professional: ['Psychologist consultation recommended', 'Stress management program enrollment', 'Consider urgent clinical review if symptoms severe']
        };
    }

    // ---------------------- Local history (localStorage) and trend rendering ----------------------
    const HISTORY_KEY = 'mhi_history_v1';

    function appendToHistory(entry) {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            const arr = raw ? JSON.parse(raw) : [];
            arr.push(entry);
            // keep last 30 entries
            while (arr.length > 30) arr.shift();
            localStorage.setItem(HISTORY_KEY, JSON.stringify(arr));
        } catch (e) {
            console.error('History save error', e);
        }
    }

    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) { return []; }
    }

    function clearHistory() {
        localStorage.removeItem(HISTORY_KEY);
        renderTrendChart();
    }

    function downloadHistoryCSV() {
        const data = loadHistory();
        if (!data.length) {
            alert('No history to download. Please make predictions first.');
            return;
        }
        
        // Prepare CSV data with proper formatting
        const header = ['Date', 'Time', 'Mental Load Index (MLI)', 'Condition'];
        const rows = data.map(r => {
            const dt = new Date(r.timestamp);
            const dateStr = dt.toLocaleDateString();
            const timeStr = dt.toLocaleTimeString();
            return [dateStr, timeStr, r.mli, r.condition].map(cell => `"${cell}"`).join(',');
        });
        
        // Combine header and rows
        const csv = [header.join(','), ...rows].join('\n');
        
        // Create blob and download
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        // Generate filename with timestamp
        const now = new Date();
        const filename = `mental_health_report_${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}.csv`;
        
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show confirmation message
        const downloadBtn = document.getElementById('downloadHistoryBtn');
        if (downloadBtn) {
            const originalText = downloadBtn.textContent;
            downloadBtn.textContent = '✓ Downloaded!';
            downloadBtn.style.backgroundColor = '#10b981';
            setTimeout(() => {
                downloadBtn.textContent = originalText;
                downloadBtn.style.backgroundColor = '';
            }, 2000);
        }
    }

    function renderTrendChart() {
        const chartDiv = document.getElementById('trendChart');
        if (!chartDiv) return;
        const data = loadHistory();
        // take last 7 days (most recent entries). If less than 7, simulate previous days by repeating first value
        const last = data.slice(-7);
        const labels = [];
        const values = [];
        if (!last.length) {
            // no data
            Plotly.purge(chartDiv);
            chartDiv.innerHTML = '<div class="empty-chart" style="padding: 40px; text-align: center; color: #888; border: 1px dashed #ccc; border-radius: 4px; background: #f9f9f9;">📊 No history yet. Make your first prediction to start tracking trends!</div>';
            return;
        }
        last.forEach(item => {
            const d = new Date(item.timestamp);
            labels.push(d.toLocaleDateString());
            values.push(item.mli);
        });
        const trace = { 
            x: labels, y: values, type: 'scatter', mode: 'lines+markers',
            line: { color: '#3b82f6', width: 3, shape: 'spline' },
            marker: { size: 8, color: values.map(v => (v <= 30) ? '#10b981' : (v <= 70) ? '#f59e0b' : '#ef4444'), opacity: 0.8, line: { color: '#fff', width: 2 } },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: '<b>MLI: %{y}</b><br>%{x}<extra></extra>'
        };
        const layout = { 
            title: { text: '📈 7-Day Mental Load Index Trend', font: { size: 16 } },
            xaxis: { title: 'Time', showgrid: true, gridcolor: '#e0e0e0' },
            yaxis: { title: 'MLI (0-100)', range: [0, 100], showgrid: true },
            height: 350,
            margin: { b: 70, l: 60, r: 40, t: 60 },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#f5f5f5'
        };
        try { Plotly.newPlot(chartDiv, [trace], layout, { responsive: true }); } catch(e){ console.error(e); }
        
        // Analyze short-term trend to provide an output sentence
        try {
            const trendOutputEl = document.getElementById('trendOutput');
            const trendInsightEl = document.getElementById('trendInsight');
            // Determine consecutive increase days from the end
            let incCount = 0, decreasingCount = 0;
            for (let i = 1; i < values.length; i++) {
                if (values[i] > values[i - 1]) incCount++;
                else if (values[i] < values[i - 1]) decreasingCount++;
            }
            const change = values[values.length - 1] - values[0];
            const pct = values.length > 1 ? Math.round((change / values[0]) * 100) : 0;
            let msg = '', insight = '';
            if (values.length === 1) {
                msg = '📊 First reading recorded. Make more to see trends.';
                insight = 'Initial baseline.';
            } else if (incCount >= decreasingCount + 2) {
                msg = `📈 INCREASING (${pct>0?'+':''}${pct}%)`;
                insight = 'Stress is rising. Apply management strategies.';
            } else if (decreasingCount >= incCount + 2) {
                msg = `📉 DECREASING (${pct}%) - Great! ✓`;
                insight = 'Excellent progress! Keep it up!';
            } else {
                msg = `➡️ STABLE at MLI ${values[values.length-1]}`;
                insight = 'Stress is consistent.';
            }
            if (trendOutputEl) trendOutputEl.innerHTML = msg;
            if (trendInsightEl) trendInsightEl.textContent = insight;
        } catch (e) {
            console.error('Trend analysis error', e);
        }
    }

    // Wire up history buttons - FIXED: moved out of nested DOMContentLoaded
    const dl = document.getElementById('downloadHistoryBtn');
    const clr = document.getElementById('clearHistoryBtn');
    if (dl) dl.addEventListener('click', downloadHistoryCSV);
    if (clr) clr.addEventListener('click', function(){ if(confirm('Clear local history?')){ clearHistory(); } });
    
    // render existing history on page load
    renderTrendChart();

    function displayProbabilityChart(probabilities) {
        const chartDiv = document.getElementById('probabilityChart');
        
        if (!chartDiv || !probabilities) return;
        
        const classes = Object.keys(probabilities || {});
        const values = (Object.values(probabilities || {})).map(v => parseFloat(v) || 0);
        
        const colors = ['#10b981', '#f59e0b', '#ec4899', '#ef4444'];
        
        const trace = {
            x: classes,
            y: values,
            type: 'bar',
            marker: {
                color: classes.map((_, i) => colors[i % colors.length])
            }
        };

        const layout = {
            title: 'Stress Level Probability Distribution',
            xaxis: { title: 'Stress Level' },
            yaxis: { title: 'Probability (%)' },
            height: 400,
            margin: { b: 100 }
        };

        try {
            Plotly.newPlot(chartDiv, [trace], layout, { responsive: true });
        } catch (e) {
            console.error('Chart error:', e);
        }
    }

    function updateTrendChart(result) {
        // This would typically fetch historical data from /api/trends
        // For now, show current data
        const trendChart = document.getElementById('trendChart');
        
        if (!trendChart) return;
        
        const x = ['Current'];
        
        // Handle both simple and complex response formats
        let stressScore = 50;
        if (result.stress_level) {
            // Simple format: convert stress level to numeric score
            const levelMap = {
                'Low': 25,
                'Moderate-Low': 50,
                'Moderate-High': 75,
                'High': 100
            };
            stressScore = levelMap[result.stress_level] || result.confidence || 50;
        } else if (result.mental_load_index) {
            // Complex format
            stressScore = result.mental_load_index.score || 50;
        }
        
        const trace = {
            x: x,
            y: [stressScore],
            type: 'scatter',
            mode: 'lines+markers',
            marker: {
                color: '#3498db',
                size: 10
            },
            line: {
                color: '#3498db',
                width: 2
            }
        };

        const layout = {
            title: 'Stress Level (Current Reading)',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Score (0-100)' },
            height: 400
        };

        Plotly.newPlot(trendChart, [trace], layout, { responsive: true });

        // Update trend insight
        const trendInsight = document.getElementById('trendInsight');
        if (trendInsight) {
            trendInsight.textContent = 'Current assessment recorded. Historical trends will populate after multiple readings.';
        }
    }

    function showLoading(show) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = show ? 'flex' : 'none';
        }
    }

    function showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        if (errorMessage) {
            errorMessage.textContent = '❌ Error: ' + message;
            errorMessage.style.display = 'block';
        } else {
            // Create error element if it doesn't exist
            const errorDiv = document.createElement('div');
            errorDiv.id = 'errorMessage';
            errorDiv.className = 'error-message';
            errorDiv.style.cssText = 'background: #fee; color: #c33; padding: 15px; border-radius: 4px; margin: 15px 0; border: 1px solid #fcc;';
            errorDiv.textContent = '❌ Error: ' + message;
            document.querySelector('main').insertBefore(errorDiv, document.querySelector('main').firstChild);
        }
    }

    function hideError() {
        const errorMessage = document.getElementById('errorMessage');
        if (errorMessage) {
            errorMessage.style.display = 'none';
        }
    }

    function getOrCreateUserId() {
        let userId = localStorage.getItem('mentalHealthUserId');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('mentalHealthUserId', userId);
        }
        return userId;
    }

    // Initialize app
    console.log('✓ Mental Health Monitoring System loaded');
});
