/**
 * VaakSetu — TTS Player
 * Calls the Flask TTS endpoint and plays the returned audio.
 */
window.ttsPlayer = (() => {
    return {
        speak: async (text, lang = 'en-IN') => {
            try {
                const res = await fetch('/api/tts/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, lang }),
                });

                if (!res.ok) throw new Error('TTS request failed');

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const audioPlayer = document.getElementById('audio-player');

                if (audioPlayer) {
                    audioPlayer.src = url;
                    audioPlayer.play();
                } else {
                    const tempAudio = new Audio(url);
                    tempAudio.play();
                }

                window.showToast?.('Playing audio summary…', 'info');
            } catch (e) {
                console.error('[ttsPlayer] Failed:', e);
                window.showToast?.('Failed to synthesize speech.', 'error');
            }
        },
    };
})();
