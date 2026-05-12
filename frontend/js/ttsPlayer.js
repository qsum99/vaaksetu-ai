window.ttsPlayer = (() => {
    return {
        speak: async (text) => {
            try {
                const res = await fetch(`http://localhost:8000/tts/speak`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, lang: 'en-IN' })
                });

                if (!res.ok) throw new Error("TTS failed");
                
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
            } catch (e) {
                console.error("Failed to play TTS", e);
                alert("Failed to synthesize speech.");
            }
        }
    };
})();
