import azure.cognitiveservices.speech as speechsdk
from config.settings import settings
from typing import Optional
import io
import os
import tempfile

class AzureSpeechService:
    
    def __init__(self):
        """Initialize Azure Speech configuration"""
        self.speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        
        # You can change the voice name here if needed
        self.speech_config.speech_synthesis_voice_name = "en-GB-SoniaNeural"
        self.speech_config.speech_recognition_language = "en-GB"

    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        try:
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            result = speech_synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"❌ TTS canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"   Error details: {cancellation.error_details}")
                return None
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return None
    
    
    def text_to_speech_stream(self, text: str):
        audio_bytes = self.text_to_speech(text)
        if audio_bytes:
            return io.BytesIO(audio_bytes)
        return None
    
    
    def speech_to_text_from_mic(self) -> Optional[str]:
        try:
            audio_config = speechsdk.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )

            speech_recognizer.properties.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, 
            "3000" 
            )
        
            speech_recognizer.properties.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "6000"  
            )
            
            print("🎤 Listening... (speak now)")
            
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print(f"✅ Recognized: {result.text}")
                return result.text
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("❌ No speech recognized")
                return None
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"❌ Recognition canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"   Error details: {cancellation.error_details}")
                return None
        
        except Exception as e:
            print(f"❌ STT Error: {e}")
            return None
    
    
    def speech_to_text_from_audio_data(self, audio_bytes: bytes) -> Optional[str]:
        """
        Convert audio bytes to text (for uploaded audio files)
        """
        # 1. Check if audio data is valid/empty
        if not audio_bytes or len(audio_bytes) < 4000:
            print("⚠️ Audio too short or empty. Please speak longer.")
            return None

        import tempfile
        import os
        
        tmp_filename = None
        
        try:
            # 2. Write bytes to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_filename = tmp_file.name
            
            # 3. Configure Azure to read from that file
            audio_config = speechsdk.audio.AudioConfig(filename=tmp_filename)
            
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 4. Perform recognition
            result = speech_recognizer.recognize_once_async().get()
            
            # 5. CRITICAL: Force clean up of Azure objects to release the file lock
            del speech_recognizer
            del audio_config
            
            # 6. Process results
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("❌ No speech could be recognized (Timeout/Silence)")
                return None
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"❌ Recognition canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"   Error details: {cancellation.error_details}")
                return None
        
        except Exception as e:
            print(f"❌ STT from audio data error: {e}")
            return None
            
        finally:
            # 7. Safe deletion - if file is still locked, don't crash the app
            if tmp_filename and os.path.exists(tmp_filename):
                try:
                    os.remove(tmp_filename)
                except PermissionError:
                    print(f"⚠️ Could not delete temp file {tmp_filename} (file in use), skipping.")
                except Exception as e:
                    print(f"⚠️ Error deleting temp file: {e}")
        
        return None
    
    
    def get_available_voices(self) -> list:
        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            result = synthesizer.get_voices_async().get()
            
            if result.reason == speechsdk.ResultReason.VoicesListRetrieved:
                en_voices = [
                    voice.short_name for voice in result.voices 
                    if voice.locale.startswith('en-')
                ]
                return en_voices
            
            return []
        
        except Exception as e:
            print(f"❌ Error getting voices: {e}")
            return []


# Standalone helper functions
def speak(text: str) -> Optional[bytes]:
    service = AzureSpeechService()
    return service.text_to_speech(text)


def listen() -> Optional[str]:
    service = AzureSpeechService()
    return service.speech_to_text_from_mic()