import { useState, useCallback, useEffect } from 'react';
import { verifyWithAI, AIVerificationResult } from '@/lib/openrouter';

interface Model {
  name: string;
  description: string;
}

interface ModelsResponse {
  models: {
    [key: string]: Model;
  };
  current: string;
}

export function useFactCheck() {
  const [currentModel, setCurrentModel] = useState('gradient_boosting');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<{[key: string]: Model}>({});
  const [loadingModels, setLoadingModels] = useState(false);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      setLoadingModels(true);
      try {
        const response = await fetch('http://localhost:5000/models', {
          // Add a timeout to avoid hanging if server is down
          signal: AbortSignal.timeout(5000)
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.status}`);
        }
        
        const data: ModelsResponse = await response.json();
        setAvailableModels(data.models);
        setCurrentModel(data.current);
      } catch (err: any) {
        console.error('Error fetching models:', err);
        setError('Failed to load available models. Using default model.');
        
        // Set default models if backend is not available
        setAvailableModels({
          "gradient_boosting": {
            "name": "Gradient Boosting",
            "description": "Traditional ML model with high accuracy on structured features"
          },
          "lstm": {
            "name": "LSTM",
            "description": "Deep learning model good at capturing sequential patterns"
          },
          "cnn": {
            "name": "CNN",
            "description": "Deep learning model effective at capturing local patterns"
          }
        });
      } finally {
        setLoadingModels(false);
      }
    };
    
    fetchModels();
  }, []);

  const predict = useCallback(async (text: string, modelId?: string) => {
    setLoading(true);
    setError(null);

    try {
      const modelToUse = modelId || currentModel;
      
      // Variables to store prediction results
      let modelPrediction = 0.5; // Default to 0.5 (uncertain)
      let modelName = modelToUse;
      
      try {
        // First, get the model prediction from the backend
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            text,
            model: modelToUse
          }),
          // Add a timeout to avoid hanging if server is down
          signal: AbortSignal.timeout(10000)
        });
  
        if (!response.ok) {
          // Try to get error details from response
          try {
            const errorData = await response.json();
            throw new Error(errorData.error || `Request failed with status ${response.status}`);
          } catch (jsonError) {
            throw new Error(`Request failed with status ${response.status}`);
          }
        }
  
        const data = await response.json();
        
        // Update current model if it changed
        if (modelToUse !== currentModel) {
          setCurrentModel(modelToUse);
        }
        
        // Store the prediction results
        modelPrediction = data.prediction;
        modelName = data.model;
      } catch (backendError) {
        console.error("Backend error:", backendError);
        setError("Backend server not available. Using AI verification only.");
      }
      
      // Now, get AI verification directly from OpenRouter
      let aiVerification: AIVerificationResult | undefined;
      
      if (text.length < 4000) // Limit text length to avoid token limits
      {
        try {
          console.log("Starting AI verification...");
          aiVerification = await verifyWithAI(text);
          console.log("AI verification result:", aiVerification);
        } catch (aiError) {
          console.error("Error during AI verification:", aiError);
          aiVerification = {
            success: false,
            error: "Error during AI verification"
          };
        }
      } else {
        aiVerification = {
          success: false,
          error: "Text too long for AI verification"
        };
      }
      
      // Determine the label based on the prediction and AI verification
      let isFake = modelPrediction > 0.5;
      
      // If Mistral says it's real (not fake) but our model says it's fake,
      // prioritize Mistral's prediction and override the model prediction
      if (aiVerification?.success && aiVerification.is_fake === false && isFake === true) {
        console.log("Mistral says REAL but model says FAKE - prioritizing Mistral's prediction");
        isFake = false; // Override to match Mistral's "real" prediction
      }
      
      const label = isFake ? 'Fake' : 'Real';
      
      return {
        probability: isFake ? modelPrediction : 1 - modelPrediction, // Adjust probability to match the final label
        label: label,
        modelName: isFake ? modelName : "Mistral (OpenRouter)", // Show which model made the final decision
        aiVerification: aiVerification,
        overridden: aiVerification?.success && aiVerification.is_fake === false && modelPrediction > 0.5
      };
    } catch (err: any) {
      setError(err.message || 'An error occurred during prediction.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [currentModel]);

  const changeModel = useCallback((modelId: string) => {
    if (availableModels[modelId]) {
      setCurrentModel(modelId);
      return true;
    }
    return false;
  }, [availableModels]);

  return { 
    loading, 
    error, 
    predict, 
    availableModels, 
    currentModel, 
    changeModel,
    loadingModels
  };
};
