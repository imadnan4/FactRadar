import React from 'react';

/**
 * Format the reasoning text from AI verification to have a clearer structure
 * @param reasoning The reasoning text from AI verification
 * @returns Formatted JSX with verdict and bullet points
 */
export const formatReasoning = (reasoning: string) => {
  if (!reasoning) return null;
  
  // Check if the text already has bullet points
  const hasBulletPoints = /[-•*]|\d+\./.test(reasoning);
  
  if (!hasBulletPoints) {
    // If no bullet points, try to split by sentences for the first 2-3 sentences
    const sentences = reasoning.split(/(?<=[.!?])\s+/);
    
    if (sentences.length <= 1) {
      return <p>{reasoning}</p>;
    }
    
    // First sentence is the verdict
    const verdict = sentences[0].trim();
    
    // Next 2-3 sentences as key points
    const keyPoints = sentences.slice(1, 4);
    
    return (
      <>
        <p className="font-medium mb-2">{verdict}</p>
        <ul className="list-disc pl-5 space-y-1.5">
          {keyPoints.map((point, index) => (
            <li key={index} className="text-white/70">{point.trim()}</li>
          ))}
        </ul>
      </>
    );
  }
  
  // If there are bullet points, split by them
  const parts = reasoning.split(/[-•*]|\d+\./);
  
  // First part is usually the verdict/summary
  const verdict = parts[0].trim();
  
  // The rest are bullet points
  const bulletPoints = parts.slice(1).filter(point => point.trim().length > 0);
  
  return (
    <>
      <p className="font-medium mb-2">{verdict}</p>
      <ul className="list-disc pl-5 space-y-1.5">
        {bulletPoints.map((point, index) => (
          <li key={index} className="text-white/70">{point.trim()}</li>
        ))}
      </ul>
    </>
  );
};