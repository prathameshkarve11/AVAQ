from kokoro import KPipeline
import soundfile as sf

print("Loading Kokoro...")

pipeline = KPipeline(lang_code="a")

text = "Hello! I am AVAQ. How can I help you today?"

print("Generating voice...")

generator = pipeline(
    text,
    voice="af_sarah",
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"avaq_voice_{i}.wav", audio, 24000)

print("Done! Voice file created.")