// OpenRouter API client for fact-checking

// The API key should be stored in an environment variable
// For client-side usage, we need to prefix it with NEXT_PUBLIC_
const OPENROUTER_API_KEY = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;

export interface AIVerificationResult {
  success: boolean;
  is_fake?: boolean;
  confidence?: number;
  reasoning?: string;
  error?: string;
}

/**
 * Cross-verify text with Mistral AI via OpenRouter
 */
export async function verifyWithAI(text: string): Promise<AIVerificationResult> {
  if (!OPENROUTER_API_KEY) {
    console.warn("OpenRouter API key not configured");
    return {
      success: false,
      error: "OpenRouter API key not configured"
    };
  }

  try {
    // Prepare the prompt for fact-checking with structured output
    const prompt = `
      You are a fact-checking expert. Please analyze the following text carefully to determine if it contains misinformation, fake news, or misleading content.

      Consider these factors in your analysis:
      - Presence of verifiable facts vs. unsubstantiated claims
      - Emotional language or sensationalism
      - Logical inconsistencies or contradictions
      - Source credibility indicators (if mentioned)
      - Scientific consensus (for scientific claims)
      - Political bias or agenda-driven content

      Text to analyze:
      "${text}"

      Respond with a JSON object containing:
      1. "is_fake": boolean (true if the text appears to contain significant misinformation, false if it appears to be generally reliable)
      2. "confidence": number between 0 and 1 indicating your confidence level in this assessment
      3. "reasoning": Your analysis should be structured as follows:
         - First sentence: Clear verdict statement (e.g., "This statement appears to be factual/misleading/opinion")
         - Then 2-3 key points in bullet form, each 1-2 sentences maximum
         - Keep the entire reasoning concise and focused on the most important factors

      Respond only with the JSON object.
    `;

    // Make the API call
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
        "HTTP-Referer": "https://factradarchecker.com", // Site URL for rankings
        "X-Title": "FactRadar", // Site title for rankings
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [
          {
            "role": "system", 
            "content": "You are a fact-checking assistant. Analyze text for misinformation and respond in JSON format."
          },
          {
            "role": "user",
            "content": prompt
          }
        ],
        "response_format": { "type": "json_object" }
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the AI's response
    const aiResponse = data.choices?.[0]?.message?.content;
    
    if (!aiResponse) {
      throw new Error("No response content from AI");
    }

    try {
      // Parse the JSON response
      const aiAnalysis = JSON.parse(aiResponse);
      
      return {
        success: true,
        is_fake: aiAnalysis.is_fake,
        confidence: aiAnalysis.confidence,
        reasoning: aiAnalysis.reasoning
      };
    } catch (jsonError) {
      console.error("Failed to parse AI response as JSON:", aiResponse);
      return {
        success: false,
        error: "Failed to parse AI response as JSON"
      };
    }
  } catch (error: any) {
    console.error("Error verifying with AI:", error);
    return {
      success: false,
      error: error.message || "An error occurred during AI verification"
    };
  }
}