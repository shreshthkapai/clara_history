# test_speech.py

from services.azure_speech import AzureSpeechService
import time

def test_speech_service():
    
    print("🧪 Testing Azure Speech Service\n")
    print("=" * 60)
    
    service = AzureSpeechService()
    print("✅ Speech service initialized")
    print(f"   Voice: {service.speech_config.speech_synthesis_voice_name}")
    print(f"   Language: {service.speech_config.speech_recognition_language}\n")
    
    print("=" * 60)
    print("TEST 1: Text-to-Speech (TTS)")
    print("=" * 60)
    
    test_text = "Hello! I'm Clara. What was the main reason for your consultation?"
    print(f"\n📝 Text to speak: \"{test_text}\"\n")
    
    print("🔊 Converting to speech...")
    audio_bytes = service.text_to_speech(test_text)
    
    if audio_bytes:
        print(f"✅ TTS successful!")
        print(f"   Audio size: {len(audio_bytes)} bytes")
        print(f"   Duration: ~{len(audio_bytes) / 32000:.1f} seconds (estimated)\n")
        
        # Save to file for manual verification
        with open("test_clara_voice.wav", "wb") as f:
            f.write(audio_bytes)
        print("💾 Saved audio to: test_clara_voice.wav")
        print("   ▶️  You can play this file to hear Clara's voice!\n")
    else:
        print("❌ TTS failed\n")
    
    print("=" * 60)
    print("TEST 2: Speech-to-Text (STT)")
    print("=" * 60)
    
    print("\n⚠️  This test requires you to speak into your microphone.")
    user_ready = input("Ready to test? Press ENTER to start, or 'skip' to skip: ")
    
    if user_ready.lower() != 'skip':
        print("\n🎤 Get ready to speak in 2 seconds...")
        time.sleep(2)
        
        recognized_text = service.speech_to_text_from_mic()
        
        if recognized_text:
            print(f"\n✅ STT successful!")
            print(f"   You said: \"{recognized_text}\"\n")
        else:
            print("\n❌ STT failed or no speech detected\n")
    else:
        print("\n⏭️  Skipped STT test\n")
    
    # Test 3: List available voices
    print("=" * 60)
    print("TEST 3: Available Voices")
    print("=" * 60)
    
    print("\n🔍 Fetching available English voices...")
    voices = service.get_available_voices()
    
    if voices:
        print(f"\n✅ Found {len(voices)} English voices:")
        # Show first 10
        for i, voice in enumerate(voices[:10], 1):
            print(f"   {i}. {voice}")
        if len(voices) > 10:
            print(f"   ... and {len(voices) - 10} more")
    else:
        print("\n❌ Could not fetch voices")
    
    print("\n" + "=" * 60)
    print("✅ SPEECH TESTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_speech_service()