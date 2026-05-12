/**
 * VaakSetu — API Client
 * Handles all REST API calls to the Flask backend.
 * Uses relative URLs since Flask serves everything.
 */
window.apiClient = {
    /**
     * Save a clinical report.
     */
    saveReport: async (data) => {
        try {
            const res = await fetch('/api/report/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (e) {
            console.error('[apiClient] saveReport failed:', e);
            window.showToast?.('Failed to save report: ' + e.message, 'error');
            return null;
        }
    },

    /**
     * Get a single report by ID.
     */
    getReport: async (id) => {
        try {
            const res = await fetch(`/api/report/${id}`);
            if (!res.ok) return null;
            return await res.json();
        } catch (e) {
            console.error('[apiClient] getReport failed:', e);
            return null;
        }
    },

    /**
     * List all reports.
     */
    listReports: async () => {
        try {
            const res = await fetch('/api/report/list');
            if (!res.ok) return [];
            return await res.json();
        } catch (e) {
            console.error('[apiClient] listReports failed:', e);
            return [];
        }
    },

    /**
     * List all patients.
     */
    getHistory: async () => {
        try {
            const res = await fetch('/api/patient/list');
            if (!res.ok) return [];
            return await res.json();
        } catch (e) {
            console.error('[apiClient] getHistory failed:', e);
            return [];
        }
    },

    /**
     * Search patients by name or village.
     */
    searchPatients: async (q) => {
        try {
            const res = await fetch(`/api/patient/search?q=${encodeURIComponent(q)}`);
            if (!res.ok) return [];
            return await res.json();
        } catch (e) {
            console.error('[apiClient] searchPatients failed:', e);
            return [];
        }
    },
};
