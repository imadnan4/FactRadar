'use client';

import React, { useState } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';

export function FakeNewsDetector() {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleDetect = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }

    setPrediction(null);
    setError(null);
    setLoading(true);
    setProgress(0);

    try {
      // Step 1: Preprocessing (now handled by backend)
      setProgress(20);
      
      // Step 2: Send to API
      setProgress(40);
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      // Step 3: Handle response
      setProgress(80);
      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
      
      const result = await response.json();
      if (result.error) {
        throw new Error(result.error);
      }
      
      setPrediction(result.prediction);
      setProgress(100);
    } catch (err: any) {
      console.error('Error during prediction:', err);
      setError(`Prediction failed: ${err.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionMessage = () => {
    if (prediction === null) return null;
    const confidence = prediction >= 0.5 ? prediction : 1 - prediction;
    const percentage = (confidence * 100).toFixed(2);
    
    return (
      <div className="mt-6 p-5 border rounded-lg bg-white shadow-sm dark:bg-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-bold">Analysis Result</h3>
          <div className={`px-3 py-1 rounded-full text-xs font-semibold ${prediction >= 0.5 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
            {prediction >= 0.5 ? 'SUSPICIOUS' : 'CREDIBLE'}
          </div>
        </div>
        
        <div className="mb-4">
          <p className="text-gray-700 dark:text-gray-300">
            {prediction >= 0.5
              ? `Our analysis indicates this content has characteristics of misinformation (${percentage}% confidence).`
              : `This content appears to be credible based on our analysis (${percentage}% confidence).`}
          </p>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mb-1">
          <div
            className={`h-2.5 rounded-full ${prediction >= 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500">
          <span>Low Risk</span>
          <span>Medium Risk</span>
          <span>High Risk</span>
        </div>
        
        <div className="mt-4 text-sm text-gray-500 border-t pt-3">
          <p>This result is based on AI analysis of text patterns. Always verify with trusted sources.</p>
        </div>
      </div>
    );
  };

  return (
    <Card className="w-full max-w-2xl mx-auto mt-8">
      <CardHeader>
        <CardTitle>FactRadar - Fake News Detector</CardTitle>
        <CardDescription>Enter a news article or text to analyze its authenticity using our AI model</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid w-full gap-4">
          <div className="space-y-2">
            <Label htmlFor="news-text">News Text</Label>
            <Textarea
              id="news-text"
              placeholder="Paste news article or text snippet here..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={10}
              disabled={loading}
            />
          </div>
          
          <div className="flex flex-col gap-2">
            <Button
              onClick={handleDetect}
              disabled={loading}
              className="w-full py-6 text-lg"
            >
              {loading ? 'Analyzing Content...' : 'Detect Fake News'}
            </Button>
            
            {loading && (
              <div className="w-full mt-2">
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-gray-700">
                    {progress < 40 ? 'Sending to server...' :
                     progress < 70 ? 'Processing text...' :
                     'Evaluating content...'}
                  </span>
                  <span className="text-sm font-medium text-gray-700">{progress}%</span>
                </div>
                <Progress value={progress} className="w-full h-3" />
              </div>
            )}
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700 font-medium">⚠️ Error: {error}</p>
            </div>
          )}
          
          {prediction !== null && getPredictionMessage()}
          
          <div className="mt-6 text-xs text-gray-500 border-t pt-3">
            <p>FactRadar analyzes text patterns using machine learning to detect potential misinformation. Results should be verified with credible sources.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
