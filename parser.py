from google import genai

client = genai.Client(api_key="AIzaSyC1d2VenPb0qVom7d7WTn7M01W7IGBxnvE")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)