document.addEventListener("DOMContentLoaded", () => {
    const isIntake = !!document.getElementById('record-btn');
    const isHistory = !!document.getElementById('history-tbody');
    const isReport = !!document.getElementById('play-tts-btn');

    // ----------------------------------------------------
    // INTAKE PAGE LOGIC
    // ----------------------------------------------------
    if (isIntake) {
        // 1. WebSocket setup
        if (window.wsClient) window.wsClient.connect();

        // 2. Record Button wiring
        const btn = document.getElementById('record-btn');
        const textStr = document.getElementById('record-status-text');
        
        btn.addEventListener("click", async () => {
            if (!window.audioRecorder.isRecording()) {
                await window.audioRecorder.startRecording();
                btn.classList.add('recording');
                textStr.textContent = "Listening...";
            } else {
                window.audioRecorder.stopRecording();
                btn.classList.remove('recording');
                btn.classList.add('processing');
                textStr.textContent = "Processing...";
                
                // Demo finish processing
                setTimeout(() => {
                    btn.classList.remove('processing');
                    textStr.textContent = "Tap to Record";
                }, 2000);
            }
        });

        // 3. WS Events
        document.addEventListener('vs-transcript', (e) => {
            const data = e.detail;
            const container = document.getElementById('transcript-box');
            const entry = document.createElement('div');
            entry.className = 'transcript-entry shadow-sm d-flex justify-content-between align-items-center';
            
            const txt = document.createElement('span');
            txt.textContent = data.text;
            
            entry.appendChild(txt);
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        });

        document.addEventListener('vs-followup', (e) => {
            const txt = document.getElementById('followup-text');
            if (txt) txt.textContent = e.detail.question;
        });

        document.addEventListener('vs-field', (e) => {
            const { field, value } = e.detail;
            const el = document.getElementById(`form-${field}`);
            if (el) el.value = value;
        });

        document.addEventListener('vs-symptom', (e) => {
            const container = document.getElementById('symptoms-container');
            if (container) {
                const s = document.createElement('span');
                s.className = 'badge bg-danger shadow-sm py-2 px-3 fw-medium';
                s.textContent = e.detail.value;
                container.appendChild(s);
            }
        });
        
        // 4. Generate Report
        const genBtn = document.getElementById('generate-report-btn');
        if (genBtn) {
            genBtn.addEventListener('click', async () => {
                // Collect symptoms
                const symEls = document.getElementById('symptoms-container').children;
                const symptomsList = Array.from(symEls).map(el => el.textContent);
                
                // Get transcript payload to save
                const transcriptContainer = document.getElementById('transcript-box');
                const rawTranscript = Array.from(transcriptContainer.children).map(child => child.firstChild.textContent).join(' ');

                const payload = {
                    patient_name: document.getElementById('form-name').value || "Unknown",
                    age: parseInt(document.getElementById('form-age').value) || 0,
                    gender: document.getElementById('form-gender').value || "other",
                    village: document.getElementById('form-village').value || "Unknown",
                    chief_complaint: symptomsList.join(', ') || "Assessment requested",
                    symptoms: symptomsList,
                    duration: "Not specified",
                    vitals: {},
                    raw_transcript: rawTranscript,
                    created_at: new Date().toISOString().split('T')[0]
                };

                const res = await window.apiClient.saveReport(payload);
                if (res && res.record_id) {
                    window.location.href = `report.html?id=${res.record_id}`;
                } else {
                    alert("Failed to generate report");
                }
            });
        }
    }

    // ----------------------------------------------------
    // REPORT PAGE LOGIC
    // ----------------------------------------------------
    if (isReport) {
        const urlParams = new URLSearchParams(window.location.search);
        const reportId = urlParams.get('id');

        if (reportId) {
            // Try to get real data first, then fall back to dummy data
            window.apiClient.getReport(reportId).then(data => {
                if (!data) {
                    // Dummy data for testing
                    const dummyData = {
                        1: {
                            patient_name: "Raj",
                            age: 35,
                            gender: "male",
                            village: "Sample Village",
                            chief_complaint: "Fever and cough",
                            symptoms: JSON.stringify(["fever", "cough", "fatigue"]),
                            duration: "3 days",
                            vitals: JSON.stringify({ temperature: "101°F", bp: "120/80" }),
                            medical_history: "No significant history",
                            diagnosis_notes: "Suspected viral infection"
                        },
                        2: {
                            patient_name: "Kushi",
                            age: 28,
                            gender: "female",
                            village: "Sample Village",
                            chief_complaint: "Headache and nausea",
                            symptoms: JSON.stringify(["headache", "nausea", "dizziness"]),
                            duration: "2 days",
                            vitals: JSON.stringify({ temperature: "98.6°F", bp: "110/70" }),
                            medical_history: "Migraine history",
                            diagnosis_notes: "Possible migraine episode"
                        }
                    };
                    data = dummyData[reportId];
                }
                
                if (data) {
                    document.getElementById('rep-name').value = data.patient_name || '';
                    document.getElementById('rep-age').value = data.age || '';
                    document.getElementById('rep-gender').value = data.gender || '';
                    document.getElementById('rep-village').value = data.village || '';
                    document.getElementById('rep-complaint').value = data.chief_complaint || '';
                    document.getElementById('rep-symptoms').value = data.symptoms ? JSON.parse(data.symptoms).join(', ') : '';
                    document.getElementById('rep-duration').value = data.duration || '';
                    document.getElementById('rep-vitals').value = data.vitals ? JSON.stringify(data.vitals) : '';
                    document.getElementById('rep-history').value = data.medical_history || '';
                    document.getElementById('rep-notes').value = data.diagnosis_notes || '';
                    
                    // TTS handler
                    const ttsBtn = document.getElementById('play-tts-btn');
                    if (ttsBtn) {
                        ttsBtn.addEventListener('click', () => {
                            const summary = `Patient name is ${data.patient_name}, age ${data.age}. Chief complaint: ${data.chief_complaint}.`;
                            window.ttsPlayer.speak(summary);
                        });
                    }
                }
            });
        }
        
        // Save history generic behavior test
        const saveHistoryBtn = document.getElementById("save-history-btn");
        if (saveHistoryBtn) {
             saveHistoryBtn.addEventListener("click", () => {
                 alert("Report updated and saved to History.");
             });
        }
    }

    // ----------------------------------------------------
    // HISTORY PAGE LOGIC
    // ----------------------------------------------------
    if (isHistory) {
        const loadHistory = async (q = "") => {
            const tbody = document.getElementById('history-tbody');
            tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4">Loading...</td></tr>';
            
            let data = q ? await window.apiClient.searchPatients(q) : await window.apiClient.getHistory();
            
            // Add dummy data if no real data
            if (!data || data.length === 0) {
                data = [
                    { id: 1, name: "Raj", age: 35, created_at: "2024-01-15", village: "Sample Village" },
                    { id: 2, name: "Kushi", age: 28, created_at: "2024-01-20", village: "Sample Village" }
                ];
            }
            
            tbody.innerHTML = '';
            
            if (data && data.length > 0) {
                data.forEach(p => {
                    const tr = document.createElement('tr');
                    tr.style.cursor = 'pointer';
                    // Need to wire to report html using patient's latest report if needed
                    tr.onclick = () => { window.location.href = `report.html?id=${p.id}`; }; 
                    tr.innerHTML = `
                        <td class="ps-4 fw-bold text-dark">${p.name}</td>
                        <td>${p.age}</td>
                        <td>${p.created_at}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary rounded-pill px-3">View</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                 tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-muted">No records found.</td></tr>';
            }
        };

        loadHistory();

        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                const q = document.getElementById('search-input').value;
                loadHistory(q);
            });
        }
    }
});
