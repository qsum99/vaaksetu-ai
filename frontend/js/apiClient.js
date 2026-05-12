const API_BASE = 'http://localhost:8000';

window.apiClient = {
    saveReport: async (data) => {
        try {
            const res = await fetch(`${API_BASE}/report/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await res.json();
        } catch (e) {
            console.error("Save report string failed", e);
            return null;
        }
    },
    
    getReport: async (id) => {
        try {
            const res = await fetch(`${API_BASE}/report/${id}`);
            return await res.json();
        } catch (e) {
            console.error("Get report failed", e);
            return null;
        }
    },
    
    getHistory: async () => {
        try {
            const res = await fetch(`${API_BASE}/patient/list`);
            return await res.json();
        } catch (e) {
            console.error("Get history failed", e);
            return [];
        }
    },
    
    searchPatients: async (q) => {
        try {
            const res = await fetch(`${API_BASE}/patient/search?q=${encodeURIComponent(q)}`);
            return await res.json();
        } catch (e) {
            console.error("Search failed", e);
            return [];
        }
    }
};
