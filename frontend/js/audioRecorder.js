window.audioRecorder = (() => {
    let mediaRecorder = null;
    let isRec = false;

    return {
        startRecording: async () => {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error("Your browser does not support audio recording here. Make sure you access the site via http://localhost:3000 or have a secure connection (HTTPS).");
                }
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                // Use a webm format with opus, good for whispering
                const mime = 'audio/webm;codecs=opus';
                mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
                
                mediaRecorder.ondataavailable = async (e) => {
                    if (e.data.size > 0 && window.wsClient) {
                        const buffer = await e.data.arrayBuffer();
                        window.wsClient.sendChunk(buffer);
                    }
                };

                // Send chunk every 1.5 seconds to balance responsiveness and context
                mediaRecorder.start(1500); 
                isRec = true;
            } catch (err) {
                console.error("Failed to get mic", err);
                alert("Microphone Error: " + err.message + "\n\n(Make sure your microphone is connected and you have granted permission in the browser URL bar).");
            }
        },

        stopRecording: () => {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                isRec = false;
            }
        },

        isRecording: () => isRec
    };
})();
