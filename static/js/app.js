/**
 * VaakSetu — Main Application Logic
 * Page-specific initialization for Intake, Report, and History pages.
 * Fixes: language_mix missing, double JSON.parse, vitals formatting,
 *        actual save-to-history, toast notifications.
 */

// ── Global Toast System ─────────────────────────────────
window.showToast = (message, type = 'info') => {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ',
    };

    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
};

// ── Main Init ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    console.log('[app.js] DOM Content Loaded');
    
    const isIntake = !!document.getElementById('record-btn');
    const isHistory = !!document.getElementById('history-tbody');
    const isReport = !!document.getElementById('play-tts-btn');

    console.log('[app.js] Page detection - Intake:', isIntake, 'History:', isHistory, 'Report:', isReport);

    // ─────────────────────────────────────────────────────
    // INTAKE PAGE
    // ─────────────────────────────────────────────────────
    if (isIntake) {
        console.log('[app.js] Setting up Intake page');
        
        // 1. Connect WebSocket
        if (window.wsClient) {
            window.wsClient.connect();
        } else {
            console.error('[app.js] wsClient not available');
        }

        // 2. Record Button
        const btn = document.getElementById('record-btn');
        const statusText = document.getElementById('record-status-text');
        const recorderRing = document.getElementById('recorder-ring');
        const miniWaveform = document.getElementById('mini-waveform');
        let transcriptCount = 0;

        if (!btn) {
            console.error('[app.js] Record button not found');
            return;
        }

        btn.addEventListener('click', async () => {
            if (!window.audioRecorder) {
                console.error('[app.js] audioRecorder not available');
                window.showToast?.('Audio recorder not ready. Please refresh the page.', 'error');
                return;
            }

            if (!window.audioRecorder.isRecording()) {
                await window.audioRecorder.startRecording();
                btn.classList.add('recording');
                recorderRing?.classList.add('recording');
                miniWaveform?.classList.add('active');
                if (statusText) statusText.textContent = 'Listening…';
                window.showToast('Recording started', 'info');
            } else {
                window.audioRecorder.stopRecording();
                btn.classList.remove('recording');
                recorderRing?.classList.remove('recording');
                miniWaveform?.classList.remove('active');
                btn.classList.add('processing');
                if (statusText) statusText.textContent = 'Processing…';

                setTimeout(() => {
                    btn.classList.remove('processing');
                    if (statusText) statusText.textContent = 'Tap to Record';
                }, 2000);
                window.showToast('Recording stopped', 'success');
            }
        });

        // 3. Transcript Events
        document.addEventListener('vs-transcript', (e) => {
            const data = e.detail;
            const container = document.getElementById('transcript-box');

            // Remove empty state
            const emptyState = container.querySelector('.transcript-empty');
            if (emptyState) emptyState.remove();

            const entry = document.createElement('div');
            entry.className = 'transcript-entry';

            const txt = document.createElement('span');
            txt.textContent = data.text;

            const langBadge = document.createElement('span');
            const lang = data.lang || 'en';
            langBadge.className = `transcript-lang lang-${lang}`;
            langBadge.textContent = lang.toUpperCase();

            entry.appendChild(txt);
            entry.appendChild(langBadge);
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;

            // Update count
            transcriptCount++;
            const countEl = document.getElementById('transcript-count');
            if (countEl) countEl.textContent = `${transcriptCount} ${transcriptCount === 1 ? 'entry' : 'entries'}`;
        });

        // 4. Follow-up Events
        document.addEventListener('vs-followup', (e) => {
            const txt = document.getElementById('followup-text');
            if (txt) {
                txt.textContent = e.detail.question;
                txt.style.animation = 'none';
                txt.offsetHeight; // trigger reflow
                txt.style.animation = 'fadeIn 0.3s ease';
            }
        });

        // 5. Field Extraction Events
        document.addEventListener('vs-field', (e) => {
            const { field, value } = e.detail;
            const el = document.getElementById(`form-${field}`);
            if (el) {
                el.value = value;
                // Flash animation
                el.style.borderColor = 'var(--accent)';
                el.style.boxShadow = '0 0 0 3px rgba(var(--accent-rgb), 0.2)';
                setTimeout(() => {
                    el.style.borderColor = '';
                    el.style.boxShadow = '';
                }, 1500);
            }
        });

        // 6. Symptom Detection Events
        document.addEventListener('vs-symptom', (e) => {
            const container = document.getElementById('symptoms-container');
            if (!container) return;

            // Remove empty state
            const emptyState = container.querySelector('.symptom-empty');
            if (emptyState) emptyState.remove();

            const tag = document.createElement('span');
            tag.className = 'symptom-tag';
            tag.textContent = e.detail.value;
            container.appendChild(tag);
        });

        // 7. Generate Report
        const genBtn = document.getElementById('generate-report-btn');
        if (genBtn) {
            genBtn.addEventListener('click', async () => {
                genBtn.disabled = true;
                genBtn.innerHTML = '<span class="loading-spinner" style="width:18px;height:18px;border-width:2px"></span> Generating…';

                // Collect symptoms
                const symEls = document.querySelectorAll('#symptoms-container .symptom-tag');
                const symptomsList = Array.from(symEls).map((el) => el.textContent);

                // Get raw transcript
                const transcriptEntries = document.querySelectorAll('#transcript-box .transcript-entry span:first-child');
                const rawTranscript = Array.from(transcriptEntries).map((el) => el.textContent).join(' ');

                const payload = {
                    patient_name: document.getElementById('form-name')?.value || 'Unknown',
                    age: parseInt(document.getElementById('form-age')?.value) || 0,
                    gender: document.getElementById('form-gender')?.value || 'other',
                    village: document.getElementById('form-village')?.value || 'Unknown',
                    chief_complaint: symptomsList.join(', ') || 'Assessment requested',
                    symptoms: symptomsList,
                    duration: document.getElementById('form-duration')?.value || 'Not specified',
                    vitals: {},
                    raw_transcript: rawTranscript,
                    language_mix: ['en'],  // FIX: was missing in original
                    created_at: new Date().toISOString().split('T')[0],
                };

                const res = await window.apiClient.saveReport(payload);
                if (res && res.record_id) {
                    window.showToast('Report generated successfully!', 'success');
                    window.location.href = `/report?id=${res.record_id}`;
                } else {
                    genBtn.disabled = false;
                    genBtn.innerHTML = `
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                        </svg>
                        Generate Report`;
                }
            });
        }
    }

    // ─────────────────────────────────────────────────────
    // REPORT PAGE
    // ─────────────────────────────────────────────────────
    if (isReport) {
        const urlParams = new URLSearchParams(window.location.search);
        const reportId = urlParams.get('id');

        if (reportId) {
            window.apiClient.getReport(reportId).then((data) => {
                if (!data || data.error) {
                    // Fallback dummy data for demo
                    const dummyData = {
                        1: {
                            patient_name: 'Raj', age: 35, gender: 'male', village: 'Pilani',
                            chief_complaint: 'Fever and cough',
                            symptoms: ['fever', 'cough', 'fatigue'],
                            duration: '3 days',
                            vitals: { temperature: '101°F', bp: '120/80' },
                            medical_history: 'No significant history',
                            diagnosis_notes: 'Suspected viral infection',
                        },
                        2: {
                            patient_name: 'Kushi', age: 28, gender: 'female', village: 'Pilani',
                            chief_complaint: 'Headache and nausea',
                            symptoms: ['headache', 'nausea', 'dizziness'],
                            duration: '2 days',
                            vitals: { temperature: '98.6°F', bp: '110/70' },
                            medical_history: 'Migraine history',
                            diagnosis_notes: 'Possible migraine episode',
                        },
                    };
                    data = dummyData[reportId];
                }

                if (data) populateReport(data);
            });
        }

        function populateReport(data) {
            setVal('rep-name', data.patient_name);
            setVal('rep-age', data.age);
            setVal('rep-gender', data.gender);
            setVal('rep-village', data.village);
            setVal('rep-complaint', data.chief_complaint);
            setVal('rep-duration', data.duration);
            setVal('rep-history', data.medical_history);
            setVal('rep-notes', data.diagnosis_notes);

            // FIX: Handle symptoms properly (already an array from Flask, not a string)
            let symptoms = data.symptoms || [];
            if (typeof symptoms === 'string') {
                try { symptoms = JSON.parse(symptoms); } catch { symptoms = [symptoms]; }
            }

            // Display as tags
            const tagsContainer = document.getElementById('rep-symptoms-tags');
            const textArea = document.getElementById('rep-symptoms');
            if (tagsContainer) {
                tagsContainer.innerHTML = '';
                symptoms.forEach((s) => {
                    const tag = document.createElement('span');
                    tag.className = 'symptom-tag';
                    tag.textContent = s;
                    tagsContainer.appendChild(tag);
                });
            }
            if (textArea) textArea.value = symptoms.join(', ');

            // FIX: Format vitals as human-readable text, not raw JSON
            let vitals = data.vitals || {};
            if (typeof vitals === 'string') {
                try { vitals = JSON.parse(vitals); } catch { vitals = {}; }
            }
            const vitalsStr = Object.entries(vitals)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' | ');
            setVal('rep-vitals', vitalsStr);

            // Followup date
            if (data.followup_date) {
                setVal('rep-followup', data.followup_date);
            }

            // TTS button
            const ttsBtn = document.getElementById('play-tts-btn');
            if (ttsBtn) {
                ttsBtn.addEventListener('click', () => {
                    const summary = `Patient name is ${data.patient_name}, age ${data.age}. Chief complaint: ${data.chief_complaint}. Symptoms include ${symptoms.join(', ')}.`;
                    window.ttsPlayer?.speak(summary);
                });
            }
        }

        function setVal(id, value) {
            const el = document.getElementById(id);
            if (el && value !== null && value !== undefined) el.value = value;
        }

        // Save to History — actually saves via API
        const saveBtn = document.getElementById('save-history-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                saveBtn.disabled = true;
                saveBtn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;border-width:2px"></span> Saving…';

                // Collect current form values
                const symptomsText = document.getElementById('rep-symptoms')?.value || '';
                const payload = {
                    patient_name: document.getElementById('rep-name')?.value || 'Unknown',
                    age: parseInt(document.getElementById('rep-age')?.value) || 0,
                    gender: document.getElementById('rep-gender')?.value || 'other',
                    village: document.getElementById('rep-village')?.value || 'Unknown',
                    chief_complaint: document.getElementById('rep-complaint')?.value || '',
                    symptoms: symptomsText.split(',').map((s) => s.trim()).filter(Boolean),
                    duration: document.getElementById('rep-duration')?.value || '',
                    vitals: {},
                    medical_history: document.getElementById('rep-history')?.value || '',
                    diagnosis_notes: document.getElementById('rep-notes')?.value || '',
                    followup_date: document.getElementById('rep-followup')?.value || null,
                    raw_transcript: '',
                    language_mix: ['en'],
                    created_at: new Date().toISOString().split('T')[0],
                };

                const res = await window.apiClient.saveReport(payload);
                if (res && res.status === 'success') {
                    window.showToast('Report saved to history!', 'success');
                } else {
                    window.showToast('Failed to save report.', 'error');
                }

                saveBtn.disabled = false;
                saveBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                        <polyline points="17 21 17 13 7 13 7 21"/>
                        <polyline points="7 3 7 8 15 8"/>
                    </svg>
                    Save to History`;
            });
        }
    }

    // ─────────────────────────────────────────────────────
    // HISTORY PAGE
    // ─────────────────────────────────────────────────────
    if (isHistory) {
        const loadHistory = async (q = '') => {
            const tbody = document.getElementById('history-tbody');
            const emptyState = document.getElementById('empty-state');
            const tableEl = document.getElementById('history-table');

            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="table-loading">
                        <div class="loading-spinner"></div>
                        <span>Loading records…</span>
                    </td>
                </tr>`;

            let data = q
                ? await window.apiClient.searchPatients(q)
                : await window.apiClient.getHistory();

            tbody.innerHTML = '';

            if (!data || data.length === 0) {
                // Show empty state
                if (tableEl) tableEl.style.display = 'none';
                if (emptyState) emptyState.style.display = 'block';
                return;
            }

            if (tableEl) tableEl.style.display = '';
            if (emptyState) emptyState.style.display = 'none';

            data.forEach((p, i) => {
                const tr = document.createElement('tr');
                tr.style.animationDelay = `${i * 0.05}s`;
                tr.style.animation = 'fadeInUp 0.3s var(--ease-out) both';
                tr.onclick = () => {
                    window.location.href = `/report?id=${p.id}`;
                };
                tr.innerHTML = `
                    <td class="patient-name">${escapeHtml(p.name || '')}</td>
                    <td>${p.age || '—'}</td>
                    <td>${escapeHtml(p.village || '—')}</td>
                    <td>${p.created_at || '—'}</td>
                    <td><button class="view-btn">View</button></td>
                `;
                tbody.appendChild(tr);
            });
        };

        loadHistory();

        // Search
        const searchBtn = document.getElementById('search-btn');
        const searchInput = document.getElementById('search-input');

        searchBtn?.addEventListener('click', () => {
            loadHistory(searchInput?.value || '');
        });

        searchInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') loadHistory(searchInput.value || '');
        });
    }
});

// ── Utility ─────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
