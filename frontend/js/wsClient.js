window.wsClient = (() => {
    let ws = null;
    let reconnectTimer = null;
    let currentSessionId = 'test-session'; 

    const updateStatus = (text, statusClass) => {
        const el = document.getElementById('ws-status');
        if (el) {
            el.textContent = text;
            el.className = `badge bg-${statusClass} p-2 shadow-sm rounded-pill`;
        }
    };

    const connect = () => {
        updateStatus("Connecting...", "secondary");
        ws = new WebSocket(`ws://localhost:8000/ws/stream?session_id=${currentSessionId}`);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            console.log("WebSocket connected");
            updateStatus("Connected", "success");
            clearTimeout(reconnectTimer);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'transcript') {
                    document.dispatchEvent(new CustomEvent('vs-transcript', { detail: data }));
                } else if (data.type === 'followup') {
                    document.dispatchEvent(new CustomEvent('vs-followup', { detail: data }));
                } else if (data.type === 'field') {
                    document.dispatchEvent(new CustomEvent('vs-field', { detail: data }));
                } else if (data.type === 'symptom') {
                    document.dispatchEvent(new CustomEvent('vs-symptom', { detail: data }));
                }
            } catch (e) {
                console.error("Failed to parse WS msg", e);
            }
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            updateStatus("Disconnected", "danger");
            reconnectTimer = setTimeout(connect, 3000);
        };

        ws.onerror = (e) => {
            console.error("WebSocket error", e);
            ws.close();
        };
    };

    return {
        connect,
        sendChunk: (buffer) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(buffer);
            }
        },
        disconnect: () => {
            if (ws) ws.close();
        }
    };
})();
