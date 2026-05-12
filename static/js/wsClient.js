/**
 * VaakSetu — WebSocket Client (Socket.IO)
 * Connects to Flask-SocketIO backend for real-time audio streaming.
 */
window.wsClient = (() => {
    let socket = null;
    let currentSessionId = 'default';

    // Check if Socket.IO library is available
    const isIOAvailable = () => {
        if (typeof io === 'undefined') {
            console.warn('[wsClient] Socket.IO library not loaded yet');
            return false;
        }
        return true;
    };

    const updateStatus = (text, state) => {
        const el = document.getElementById('ws-status');
        if (!el) return;
        el.className = `connection-status ${state}`;
        const dot = el.querySelector('.status-dot');
        const txt = el.querySelector('.status-text');
        if (txt) txt.textContent = text;
    };

    const connect = () => {
        if (!isIOAvailable()) {
            console.error('[wsClient] Socket.IO not loaded. Retrying in 500ms...');
            updateStatus('Loading...', '');
            setTimeout(connect, 500);
            return;
        }

        updateStatus('Connecting…', '');
        console.log('[wsClient] Starting connection to', window.location.origin);

        try {
            // Connect to the Flask-SocketIO server
            socket = io(window.location.origin, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionDelay: 2000,
                reconnectionAttempts: 10,
            });

            socket.on('connect', () => {
                console.log('[WS] Connected');
                updateStatus('Connected', 'connected');
                // Start a session
                socket.emit('start_session', { session_id: currentSessionId });
            });

            socket.on('disconnect', () => {
                console.log('[WS] Disconnected');
                updateStatus('Disconnected', 'disconnected');
            });

            socket.on('connect_error', (err) => {
                console.error('[WS] Connection error:', err);
                updateStatus('Connection Error', 'disconnected');
            });

            // ── Event listeners for AI pipeline results ──
            socket.on('transcript', (data) => {
                document.dispatchEvent(new CustomEvent('vs-transcript', { detail: data }));
            });

            socket.on('followup', (data) => {
                document.dispatchEvent(new CustomEvent('vs-followup', { detail: data }));
            });

            socket.on('field', (data) => {
                document.dispatchEvent(new CustomEvent('vs-field', { detail: data }));
            });

            socket.on('symptom', (data) => {
                document.dispatchEvent(new CustomEvent('vs-symptom', { detail: data }));
            });

            socket.on('status', (data) => {
                console.log('[WS] Status:', data.message);
            });

            socket.on('session_started', (data) => {
                console.log('[WS] Session started:', data.session_id);
                currentSessionId = data.session_id;
            });
        } catch (err) {
            console.error('[wsClient] Failed to initialize Socket.IO:', err);
            updateStatus('Connection Failed', 'disconnected');
        }
    };

    return {
        connect,
        sendChunk: (buffer) => {
            if (socket && socket.connected) {
                socket.emit('audio_chunk', {
                    session_id: currentSessionId,
                    audio: buffer,
                });
            } else {
                console.warn('[wsClient] Socket not connected, cannot send chunk');
            }
        },
        disconnect: () => {
            if (socket) socket.disconnect();
        },
        isConnected: () => socket && socket.connected,
    };
})();
