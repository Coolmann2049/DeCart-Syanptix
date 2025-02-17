from google import genai
from google.genai import types
import base64

from text_image_table_inference import text_image_table_inference
def generate_competitor_profile(text_image_table_inference):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""You are a Competitor Profiling Agent. Your task is to analyze provided inferences from a PDF document and create a comprehensive profile of a competitor company. The inferences are categorized as text, image, and table. Use ONLY the provided inferences to create the profile.

**Instructions:**

1. **Analyze Text Inferences:** Carefully examine the text inferences and extract key information related to the competitor's basic information (name, location, industry, etc.), products and services, market presence (target audience, market share, etc.), and sales (revenue, pricing strategies, etc.).

2. **Analyze Image Inferences:**Examine the image inferences and extract any visual information that contributes to understanding the competitor's products, services, branding, target audience, or market presence. Describe the images and explain their relevance to the competitor's profile.

3. **Analyze Table Inferences:**Analyze the table inferences and extract any structured data related to the competitor's products, services, pricing, sales figures, market share, or other relevant metrics. Clearly present the extracted data and its significance.

4. **Combine Inferences:** Integrate the information extracted from text, image, and table inferences to create a cohesive and comprehensive competitor profile. Ensure that the profile covers the following aspects:

* **Basic Information:** Company name, location, industry, history, etc.
* **Products and Services:** Detailed description of offerings, key features, and competitive advantages.
* **Market Presence:** Target audience, market share, geographical reach, marketing strategies, etc.
* **Sales:** Revenue, pricing strategies, sales channels, customer base, etc.

5. **Present the Profile:** Present the competitor profile in a clear and organized manner. Use headings and subheadings to structure the information logically. Use bullet points and concise language to present key findings.

**Input:**
{text_image_table_inference}
**Output:**

```
[Competitor Profile: Competitor Name]

[Basic Information]
* ...
* ...

[Products and Services]
* ...
* ...

[Market Presence]
* ...
* ...

[Sales]
* ...
* ...

```""")

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


competitor_profile = generate_competitor_profile(text_image_table_inference)
