from google import genai
from google.genai import types
import base64
from text_image_table_inference import text_image_table_inference
def generate_competitive_strategy(text_image_table_inference):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""You are a competitive strategy manager. Your task is to analyze competitor's potential future strategies based on the provided inferences. Rely strongly on these inferences.

Inferences:
{text_image_table_inference}
Instructions:

    *  Clearly and concisely describe the competitor's likely future strategy. Be specific, outlining the key actions they are likely to take.
    * Analyze the potential impact of this predicted strategy on your own business.  Consider both the threats and opportunities it presents.  Be as detailed as possible in your analysis.
    *  Propose potential counter-strategies a business could employ to capitalize on the opportunities presented by the competitor's predicted strategy.  Again, be specific and actionable.""")

  model = "gemini-2.0-pro-exp-02-05"
  contents = [
    types.Content(
      role="user",
      parts=[
        text1
      ]
    )
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    seed = 0,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  generated_text = ""
  for chunk in client.models.generate_content_stream(
          model=model,
          contents=contents,
          config=generate_content_config,
  ):
      generated_text += chunk.text
  return generated_text

competitive_strategy = generate_competitive_strategy(text_image_table_inference)

