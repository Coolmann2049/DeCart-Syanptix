from google import genai
from google.genai import types
from text_image_table_inference import text_image_table_inference

def generate_summary(text_image_table_inference):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""You are a financial analyst tasked with summarizing a financial document based *solely* on provided inferences.  Your summary should not include any information directly from the document itself, only what can be derived from the inferences.

**Instructions:**

1. Combine the interpreted inferences into a single, detailed summary of the financial document.  Ensure the summary reflects the combined understanding derived from all inferences.
2. Do not include any extra information in your summary.  Focus only on the provided inferences.

**Inferences:**

{text_image_table_inference}

**Summary:**""")

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
  )

  generated_text = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
      generated_text += chunk.text
  return generated_text
