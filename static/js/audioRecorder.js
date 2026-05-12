/**
 * VaakSetu — Audio Recorder
 * Captures microphone audio and sends chunks via WebSocket.
 */
window.audioRecorder = (() => {
    let mediaRecorder = null;
    let isRec = false;

    // Check if browser supports audio recording
    const isSupported = () => {
        const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        console.log('[audioRecorder] Browser support check:', supported);
        return supported;
    };

    return {
        startRecording: async () => {
            try {
                // Check browser support
                if (!isSupported()) {
                    throw new Error(
                        'Your browser does not support audio recording. ' +
                        'Please use Chrome, Firefox, Edge, or Safari on a secure connection (HTTPS or localhost).'
                    );
                }

                // Request microphone permission
                console.log('[audioRecorder] Requesting microphone permission...');
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });

                console.log('[audioRecorder] Microphone access granted');

                // Find best audio format
                const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm';

                mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
                console.log('[audioRecorder] MediaRecorder created with mime:', mime);

                mediaRecorder.ondataavailable = async (e) => {
                    if (e.data.size > 0) {
                        if (window.wsClient && window.wsClient.isConnected?.()) {
                            const buffer = await e.data.arrayBuffer();
                            window.wsClient.sendChunk(buffer);
                            console.log('[audioRecorder] Sent audio chunk:', e.data.size, 'bytes');
                        } else {
                            console.warn('[audioRecorder] WebSocket not connected, buffering audio');
                        }
                    }
                };

                mediaRecorder.onerror = (event) => {
                    console.error('[audioRecorder] Error:', event.error);
                    window.showToast?.('Recording Error: ' + event.error, 'error');
                };

                // Send a chunk every 1.5 seconds
                mediaRecorder.start(1500);
                isRec = true;
                console.log('[audioRecorder] Recording started');
            } catch (err) {
                console.error('[audioRecorder] Failed to start:', err.name, err.message);
                
                // Provide specific error messages
                let userMessage = err.message;
                if (err.name === 'NotAllowedError') {
                    userMessage = 'Microphone permission denied. Please allow access in browser settings.';
                } else if (err.name === 'NotFoundError') {
                    userMessage = 'No microphone found. Please check your device.';
                } else if (err.name === 'NotSupportedError') {
                    userMessage = 'Audio recording not supported in this browser.';
                }
                
                window.showToast?.(userMessage, 'error');
            }
        },

        stopRecording: () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach((t) => t.stop());
                isRec = false;
                console.log('[audioRecorder] Recording stopped');
            }
        },

        isRecording: () => isRec,
    };
})();
