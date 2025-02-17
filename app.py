from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from google import genai
from google.genai import types

def generate_initial_inference(pdf_path):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )
  with open(pdf_path, "rb") as f:  # Open local file in binary read mode
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

def generate_swot_analysis(text_image_table_inference):
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

def generate_key_analysis(swot_analysis,competitive_strategy,competitor_profile):
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text=f"""As a Competitive Analyst, your task is to generate a detailed key analysis of a competitor based on the provided SWOT analysis, competitor profile, and competitor strategy.  Ensure you incorporate all three provided analyses into your assessment.  Be specific and detailed in your approach.

Output your analysis in the following format:

1. **Executive Summary:** Briefly summarize the key findings of your competitor analysis.

2. **SWOT Analysis Breakdown:**  Provide a detailed breakdown of the competitor's strengths, weaknesses, opportunities, and threats. Explain how each element impacts their competitive position.  Reference the provided SWOT analysis: 
{swot_analysis}
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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded PDFs
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) #Create the upload folder

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file part'})

        pdf_file = request.files['pdf_file']

        if pdf_file.filename == '':
            return jsonify({'error': 'No selected file'})

        if pdf_file and allowed_file(pdf_file.filename): #Check for allowed file types
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(pdf_path) #Save the file locally


            text_image_table_inference = generate_initial_inference(pdf_path)
            if text_image_table_inference:
                summary = generate_summary(text_image_table_inference)
                swot_analysis = generate_swot_analysis(text_image_table_inference)
                competitive_strategy = generate_competitive_strategy(text_image_table_inference)
                competitor_profile = generate_competitor_profile(text_image_table_inference)
                key_analysis = generate_key_analysis(swot_analysis,competitive_strategy,competitor_profile)
                if summary and key_analysis:
                    return jsonify({'summary': summary,
                                    'key_analysis': key_analysis })
                else:
                    return jsonify({'error': 'Gemini call failed'})
            else:
                return jsonify({'error': 'PDF extraction failed'})
        else:
            return jsonify({'error': 'File type not allowed'})

    return render_template("index.html")

ALLOWED_EXTENSIONS = {'pdf'} #Allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>') #Route to serve uploaded files
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)