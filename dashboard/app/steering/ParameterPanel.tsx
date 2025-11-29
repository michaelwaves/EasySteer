'use client';

import { SteeringConfig } from './SteeringComparison';
import { Loader2 } from 'lucide-react';

// Emotion vectors mapping
const EMOTION_VECTORS = {
  happiness: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt',
  sadness: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/sadness_response_avg_diff.pt',
  disgust: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/disgust_response_avg_diff.pt',
  anger: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/anger_response_avg_diff.pt',
  fear: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/fear_response_avg_diff.pt',
  surprise: 'vectors/persona_vectors/Qwen2.5-7B-Instruct/surprise_response_avg_diff.pt'
};

const EMOTION_LABELS = {
  happiness: 'Happiness 😃',
  sadness: 'Sadness 😭',
  disgust: 'Disgust 🤮',
  anger: 'Anger 😡',
  fear: 'Fear 😨',
  surprise: 'Surprise 😮'
};

interface ParameterPanelProps {
  config: SteeringConfig;
  onConfigChange: (config: SteeringConfig) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export function ParameterPanel({
  config,
  onConfigChange,
  onSubmit,
  isLoading
}: ParameterPanelProps) {
  const handleChange = (key: keyof SteeringConfig, value: any) => {
    onConfigChange({
      ...config,
      [key]: value
    });
  };

  return (
    <div className="w-80 h-full flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <h2 className="text-base font-semibold text-gray-900">Parameters</h2>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {/* Message Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Message
          </label>
          <textarea
            value={config.message}
            onChange={(e) => handleChange('message', e.target.value)}
            disabled={isLoading}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-400 disabled:bg-gray-50 disabled:cursor-not-allowed resize-none"
            rows={4}
          />
        </div>

        {/* Model Path */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Model Name
          </label>
          <input
            type="text"
            value={config.model_path}
            disabled={true}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-400 bg-gray-50 cursor-not-allowed text-gray-600"
          />
        </div>

        {/* Emotion Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Emotion
          </label>
          <select
            value={Object.entries(EMOTION_VECTORS).find(([_, path]) => path === config.steering_vector_path)?.[0] || 'happiness'}
            onChange={(e) => {
              const emotion = e.target.value as keyof typeof EMOTION_VECTORS;
              handleChange('steering_vector_path', EMOTION_VECTORS[emotion]);
            }}
            disabled={isLoading}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-400 disabled:bg-gray-50 disabled:cursor-not-allowed"
          >
            {Object.entries(EMOTION_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </div>

        {/* Steering Parameters */}
        <div className="space-y-4">
          <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
            Steering Controls
          </h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Scale: {config.steering_vector_scale.toFixed(2)}
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.05"
              value={config.steering_vector_scale}
              onChange={(e) => handleChange('steering_vector_scale', parseFloat(e.target.value))}
              disabled={isLoading}
              className="w-full disabled:cursor-not-allowed"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Algorithm
            </label>
            <select
              value={config.algorithm}
              onChange={(e) => handleChange('algorithm', e.target.value)}
              disabled={isLoading}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-400 disabled:bg-gray-50 disabled:cursor-not-allowed"
            >
              <option value="direct">Direct</option>
              <option value="loreft">LoReFT</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Layers (comma-separated)
            </label>
            <input
              type="text"
              value={config.target_layers}
              onChange={(e) => handleChange('target_layers', e.target.value)}
              disabled={isLoading}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-400 disabled:bg-gray-50 disabled:cursor-not-allowed font-mono text-xs"
            />
          </div>
        </div>


        {/* Generation Parameters */}
        <div className="space-y-4">
          <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
            Generation
          </h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Temperature: {config.temperature.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={config.temperature}
              onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
              disabled={isLoading}
              className="w-full disabled:cursor-not-allowed"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Tokens: {config.max_tokens}
            </label>
            <input
              type="range"
              min="1"
              max="4096"
              step="32"
              value={config.max_tokens}
              onChange={(e) => handleChange('max_tokens', parseInt(e.target.value))}
              disabled={isLoading}
              className="w-full disabled:cursor-not-allowed"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Repetition Penalty: {config.repetition_penalty.toFixed(2)}
            </label>
            <input
              type="range"
              min="1.0"
              max="2.0"
              step="0.05"
              value={config.repetition_penalty}
              onChange={(e) => handleChange('repetition_penalty', parseFloat(e.target.value))}
              disabled={isLoading}
              className="w-full disabled:cursor-not-allowed"
            />
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div className="border-t border-gray-200 px-6 py-4">
        <button
          onClick={onSubmit}
          disabled={isLoading}
          className="w-full px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded hover:bg-gray-800 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          {isLoading ? 'Generating...' : 'Generate'}
        </button>
      </div>
    </div>
  );
}
