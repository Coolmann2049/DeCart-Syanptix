from google import genai
from google.genai import types
from websockets.utils import generate_key

from  SWOT_analysis import SWOT_analysis
from competitive_strategy import competitive_strategy
from Competitor_profile import competitor_profile

def generate_key_analysis(SWOT_analysis,competitive_strategy,competitor_profile):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""As a Competitive Analyst, your task is to generate a detailed key analysis of a competitor based on the provided SWOT analysis, competitor profile, and competitor strategy.  Ensure you incorporate all three provided analyses into your assessment.  Be specific and detailed in your approach.

Output your analysis in the following format:

1. **Executive Summary:** Briefly summarize the key findings of your competitor analysis.

2. **SWOT Analysis Breakdown:**  Provide a detailed breakdown of the competitor's strengths, weaknesses, opportunities, and threats. Explain how each element impacts their competitive position.  Reference the provided SWOT analysis: 
{SWOT_analysis}
3. **Competitor Profile Analysis:** Analyze the competitor's profile, including their market positioning, target audience, product/service offerings, and key differentiators.  Relate this information to their competitive landscape. Reference the provided competitor profile: 
{competitor_profile}
4. **Competitor Strategy Analysis:**  Analyze the competitor's overall strategy, including their marketing approach, sales tactics, and product development roadmap.  Assess the effectiveness of their strategy and potential challenges. Reference the provided competitor strategy: 
{competitive_strategy}
5. **Key Competitive Advantages and Disadvantages:**  Summarize the competitor's key competitive advantages and disadvantages based on the combined insights from the SWOT analysis, competitor profile, and competitor strategy.

6. **Recommendations:** Provide actionable recommendations on how to capitalize on the competitor's weaknesses and mitigate the risks posed by their strengths.


Ensure your analysis is thorough, insightful, and data-driven.  Use the provided information to support your claims and conclusions.""")

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

key_analysis = generate_key_analysis(SWOT_analysis,competitive_strategy,competitor_profile)
