from google import genai
from google.genai import types
import base64

from text_image_table_inference import text_image_table_inference


def generate_SWOT_analysis(text_image_table_inference):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""You are a SWOT analysis master. You will be provided with inferences from a PDF. Your task is to perform a detailed SWOT analysis based on these inferences.  Please clearly separate your analysis into the following sections:

**Strengths:**
*  [List and describe the strengths identified in the inferences.]

**Weaknesses:**
* [List and describe the weaknesses identified in the inferences.]

**Opportunities:**
* [List and describe the opportunities identified in the inferences.]

**Threats:**
* [List and describe the threats identified in the inferences.]

Inferences from the PDF: {text_image_table_inference}""")
  textsi_1 = """Rely heavily on the provided inferences for the following analysis. Pretend you are a SWOT analysis master"""

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        text1
      ]
    )
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 0.9,
    top_p = 0.95,
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
    system_instruction=[types.Part.from_text(text=textsi_1)],
  )
  generated_text = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
      generated_text += chunk.text
  return generated_text

SWOT_analysis = generate_SWOT_analysis(text_image_table_inference)

