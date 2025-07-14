'use client'

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Shield, Play, Trash2, Radar, Zap, BarChart3 } from 'lucide-react'
import { useFactCheck } from '@/hooks/useFactCheck'
import { formatReasoning } from '@/lib/formatReasoning'

interface AnalysisResult {
  probability: number;
  label: string;
  aiVerification?: {
    success: boolean;
    is_fake?: boolean;
    error?: string;
    reasoning?: string; // Added reasoning field
  };
}

export default function Page() {
  const [newsInput, setNewsInput] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [currentSection, setCurrentSection] = useState('home')
  const [charCount, setCharCount] = useState(0)
  const [isScrolled, setIsScrolled] = useState(false)
  const [selectedModel, setSelectedModel] = useState('gradient_boosting')
  const [availableModels, setAvailableModels] = useState<{ [key: string]: { name: string; description: string } }>({});

  const { loading: modelLoading, error: modelError, predict, currentModel, changeModel } = useFactCheck();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Fetch models from backend on mount
  useEffect(() => {
    fetch('http://localhost:5000/models')
      .then(res => res.json())
      .then(data => {
        if (data.models) setAvailableModels(data.models);
      });
  }, [])

  useEffect(() => {
    if (currentModel && currentModel !== selectedModel) {
      setSelectedModel(currentModel);
    }
  }, [currentModel]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    setNewsInput(value)
    setCharCount(value.length)
    setAnalysisResult(null);
  }

  const handleAnalyze = async () => {
    if (newsInput.trim() && predict) {
      setIsAnalyzing(true)
      setAnalysisResult(null)
      const result = await predict(newsInput, selectedModel); // pass model
      setAnalysisResult(result);
      setIsAnalyzing(false)
    }
  }

  const trySample = () => {
    const sampleText = "Scientists from NASA have released new data showing continued warming trends across global temperature measurements. The comprehensive study, published in the Journal of Climate Science, analyzed temperature data from over 6,000 weather stations worldwide."
    setNewsInput(sampleText)
    setCharCount(sampleText.length)
    setAnalysisResult(null);
  }

  const clearText = () => {
    setNewsInput('')
    setCharCount(0)
    setAnalysisResult(null);
  }

  const renderResult = () => {
    if (isAnalyzing || modelLoading) {
      return (
        <div className="mt-6 text-center text-white/80 animate-pulse">
          {modelLoading ? 'Loading model...' : 'Analyzing...'}
        </div>
      );
    }
    
    if (modelError) {
        return (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
                <p className="font-bold">Error</p>
                <p className="text-sm">{modelError}</p>
            </div>
        )
    }

    if (!analysisResult) return null;

    const isFake = analysisResult.label === 'Fake';
    const confidence = isFake ? analysisResult.probability : 1 - analysisResult.probability;

    return (
      <div className={`mt-6 p-6 rounded-xl border ${isFake ? 'border-red-500/50' : 'border-green-500/50'} ${isFake ? 'bg-red-900/20' : 'bg-green-900/20'} backdrop-blur-xl shadow-lg`}>
        <div className="flex items-center space-x-4">
          <div className={`p-3 rounded-full ${isFake ? 'bg-red-500/30' : 'bg-green-500/30'} backdrop-blur-sm`}>
            <Shield className={`w-8 h-8 ${isFake ? 'text-red-400' : 'text-green-400'}`} />
          </div>
          <div>
            <p className={`text-2xl font-bold ${isFake ? 'text-red-300' : 'text-green-300'}`}>
              Result: {analysisResult.label}
            </p>
            <p className="text-white/70">
              Confidence: {(confidence * 100).toFixed(2)}%
            </p>
            {/* Show Mistral/OpenRouter result if present */}
            {analysisResult.aiVerification && (
              <div className="mt-2 text-xs text-white/60">
                <span className="font-semibold">OpenRouter Mistral:</span>{" "}
                {analysisResult.aiVerification.success
                  ? analysisResult.aiVerification.is_fake === false
                    ? "REAL"
                    : "FAKE"
                  : "Unavailable"}
                {analysisResult.aiVerification.error && (
                  <span className="ml-2 text-red-400">{analysisResult.aiVerification.error}</span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Show reasoning from Mistral if available */}
        {analysisResult.aiVerification && analysisResult.aiVerification.success && analysisResult.aiVerification.reasoning && (
          <div className={`mt-4 p-6 rounded-xl border ${isFake ? 'border-red-500/50' : 'border-green-500/50'} ${isFake ? 'bg-red-900/20' : 'bg-green-900/20'} backdrop-blur-xl shadow-lg`}>
            <div className="font-semibold text-blue-300 mb-2">AI Cross Verification</div>
            <div className="text-sm whitespace-pre-line text-white/90">{analysisResult.aiVerification.reasoning}</div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Aurora Background Effect */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Base gradient layer */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-cyan-500/5 to-blue-600/10 aurora-soft-pulse"></div>

        {/* Main aurora streams - horizontal flowing bands */}
        <div className="absolute top-1/4 left-0 w-full h-32 bg-gradient-to-r from-transparent via-blue-500/20 to-transparent blur-2xl aurora-flow"></div>
        <div className="absolute top-1/2 left-0 w-full h-24 bg-gradient-to-r from-transparent via-cyan-400/15 to-transparent blur-xl aurora-flow delay-1000"></div>
        <div className="absolute top-3/4 left-0 w-full h-28 bg-gradient-to-r from-transparent via-blue-400/18 to-transparent blur-2xl aurora-flow delay-2000"></div>

        {/* Large gentle aurora orbs */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/15 rounded-full blur-3xl aurora-gentle-drift"></div>
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-cyan-500/12 rounded-full blur-3xl aurora-gentle-drift delay-1500"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-blue-400/10 rounded-full blur-2xl aurora-wave-slow"></div>

        {/* Medium flowing elements */}
        <div className="absolute top-1/3 right-1/5 w-64 h-64 bg-cyan-300/8 rounded-full blur-2xl aurora-soft-pulse delay-500"></div>
        <div className="absolute bottom-1/3 left-1/5 w-56 h-56 bg-blue-300/10 rounded-full blur-xl aurora-gentle-drift delay-800"></div>

        {/* Subtle overlay gradients */}
        <div className="absolute inset-0 bg-gradient-to-t from-transparent via-blue-500/3 to-transparent aurora-wave-slow delay-300"></div>
      </div>

      {/* Header removed as requested */}

      <main className="relative z-10 max-w-4xl mx-auto px-6 py-16">
        <div className="text-center">
          <h1 className="text-5xl md:text-7xl font-extrabold text-white tracking-tight leading-tight">
            Uncover the Truth, <br/> Instantly.
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-lg text-white/70">
            FactRadar uses advanced AI to analyze news articles, social media posts, and other text sources to detect indicators of fake news. Paste your text below to get started.
          </p>
        </div>

        <div className="mt-12 p-8 bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-3xl shadow-2xl shadow-blue-500/10">
          <div className="relative">
            <textarea
              id="newsInput"
              value={newsInput}
              onChange={handleInputChange}
              placeholder="Paste the news article or text here..."
              className="w-full h-64 p-6 bg-transparent text-white/90 placeholder-white/40 text-lg rounded-xl border-2 border-white/10 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 ease-out resize-none"
              maxLength={5000}
            />
            <div className="absolute bottom-4 right-4 text-xs text-white/50">
              {charCount} / 5000
            </div>
          </div>
          <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-3">
                <Button 
                  onClick={handleAnalyze} 
                  disabled={isAnalyzing || modelLoading || !newsInput.trim()} 
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-8 h-12 text-lg rounded-full transition-all duration-300 ease-out hover:scale-105 shadow-lg hover:shadow-blue-500/25 disabled:opacity-50 disabled:scale-100 flex items-center"
                >
                  <Radar className="w-5 h-5 mr-2" />
                  {isAnalyzing || modelLoading ? (modelLoading ? 'Loading Model...' : 'Analyzing...') : 'Analyze'}
                </Button>
                
                <div className="flex items-center">
                  <select
                    id="modelDropdown"
                    value={selectedModel}
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      if (changeModel) changeModel(e.target.value);
                    }}
                    className="h-12 text-sm rounded-full border border-white/20 bg-slate-800 text-white px-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                    disabled={modelLoading}
                  >
                    {Object.entries(availableModels).map(([key, model]) => (
                      <option key={key} value={key} className="bg-slate-800 text-white">
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>
                
                <Button 
                  variant="ghost" 
                  onClick={trySample} 
                  className="text-white/60 hover:text-white hover:bg-white/10 rounded-full h-12 flex items-center"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Try Sample
                </Button>
              </div>
            </div>
            <Button variant="ghost" onClick={clearText} className="text-white/60 hover:text-white hover:bg-white/10 rounded-full">
              <Trash2 className="w-4 h-4 mr-2" />
              Clear
            </Button>
          </div>

          {renderResult()}

        </div>

        <div className="mt-24 text-center">
          <h2 className="text-3xl font-bold text-white">How It Works</h2>
          <div className="mt-8 grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="p-6 bg-white/5 rounded-xl border border-white/10">
              <Zap className="w-8 h-8 text-blue-400" />
              <h3 className="mt-4 text-xl font-semibold text-white">Real-time Analysis</h3>
              <p className="mt-2 text-white/60">Our model processes text instantly, providing immediate feedback without sending your data to a server.</p>
            </div>
            <div className="p-6 bg-white/5 rounded-xl border border-white/10">
              <BarChart3 className="w-8 h-8 text-cyan-400" />
              <h3 className="mt-4 text-xl font-semibold text-white">Advanced AI</h3>
              <p className="mt-2 text-white/60">Built on a sophisticated neural network trained on millions of articles to recognize patterns of misinformation.</p>
            </div>
            <div className="p-6 bg-white/5 rounded-xl border border-white/10">
              <Shield className="w-8 h-8 text-green-400" />
              <h3 className="mt-4 text-xl font-semibold text-white">Privacy First</h3>
              <p className="mt-2 text-white/60">Your text is analyzed directly in your browser. Nothing is ever stored or sent to us.</p>
            </div>
          </div>
        </div>

      </main>
    </div>
  )
}