from google import genai
from google.genai import types
from pyxnat.core.uriutil import file_path


def generate_initial_inference(local_pdf_path):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )
  with open(local_pdf_path, "rb") as f:  # Open local file in binary read mode
    pdf_bytes = f.read()  # Read the file content as bytes

  financial_document = types.Part.from_bytes(
    data=pdf_bytes,  # Use from_bytes with the file content
    mime_type="application/pdf",
  )

  text1 = types.Part.from_text(text="""You are a financial analyst tasked with analyzing a financial document. Your goal is to extract all text, images, and tables, and then draw meaningful inferences from each.  Please follow this structure:

**Text Understanding:**

1. **Inferences:** [Provide detailed inferences based on the extracted text. Explain the meaning and significance of the text, focusing on financial implications. Be explicit and thorough in your analysis.]

**Image Understanding:**

1. **Inferences:** [Provide detailed inferences based on each image. Explain what the images represent and their financial significance.  Connect them to the extracted text inference where possible.]

**Table Understanding:**

1. **Inferences:** [Provide detailed inferences based on each table. Analyze the data within the tables, highlighting trends, patterns, and key financial insights. Explain the meaning and significance of the data in relation to the overall financial document.]


**Financial Document:**
""")
  textsi_1 = ""
  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        text1,
        financial_document
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

text_image_table_inference = generate_initial_inference(pdf_path)