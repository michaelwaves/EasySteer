'use client';

import { useState } from 'react';
import { ParameterPanel } from './ParameterPanel';
import { ChatDisplay } from './ChatDisplay';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export interface SteeringConfig {
  message: string;
  model_path: string;
  temperature: number;
  max_tokens: number;
  repetition_penalty: number;
  steering_vector_path: string;
  steering_vector_scale: number;
  target_layers: string;
  algorithm: string;
}

export interface ComparisonResult {
  success: boolean;
  normal_response: string;
  steered_response?: string;
  config?: Record<string, any>;
  error?: string;
}

export function SteeringComparison() {
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<SteeringConfig>({
    message: "Alice's dog has passed away. Please comfort her.",
    model_path: 'Qwen/Qwen2.5-7B-Instruct',
    temperature: 0.6,
    max_tokens: 256,
    repetition_penalty: 1.1,
    steering_vector_path: '/mnt/nw/home/m.yu/repos/EasySteer/vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt',
    steering_vector_scale: 0.5,
    target_layers: '8,9,10,11,12,13,14,15,16,17,18,19,20',
    algorithm: 'direct',
  });

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

      const layers = config.target_layers
        .split(',')
        .map(l => parseInt(l.trim()))
        .filter(n => !isNaN(n));

      const requestBody = {
        message: config.message,
        history: [
          {
            role: 'system',
            content: 'you are a helpful assistant'
          }
        ],
        steered_history: [
          {
            role: 'system',
            content: 'you are a helpful assistant'
          }
        ],
        model: {
          path: config.model_path
        },
        steering_vector: {
          path: config.steering_vector_path,
          scale: config.steering_vector_scale,
          target_layers: layers,
          algorithm: config.algorithm,
          prefill_trigger_token_ids: [-1],
          generate_trigger_token_ids: [-1],
          normalize: false
        },
        gpu_devices: '0',
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        repetition_penalty: config.repetition_penalty,
        debug: false
      };

      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex w-full h-full overflow-hidden">
      {/* Parameter Panel */}
      <div
        className={`transition-all duration-300 ease-in-out border-r border-gray-200 overflow-y-auto ${
          isPanelOpen ? 'w-80' : 'w-12'
        }`}
      >
        {isPanelOpen && (
          <ParameterPanel
            config={config}
            onConfigChange={setConfig}
            onSubmit={handleSubmit}
            isLoading={isLoading}
          />
        )}
      </div>

      {/* Toggle Button */}
      <button
        onClick={() => setIsPanelOpen(!isPanelOpen)}
        className="w-12 border-r border-gray-200 flex items-center justify-center hover:bg-gray-50 transition-colors group"
        title={isPanelOpen ? 'Hide panel' : 'Show panel'}
      >
        {isPanelOpen ? (
          <ChevronLeft className="w-5 h-5 text-gray-400 group-hover:text-gray-600" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600" />
        )}
      </button>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {error && (
          <div className="bg-red-50 border-b border-red-200 px-6 py-4">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        <div className="flex-1 overflow-auto flex gap-4 p-6">
          {/* Normal Response */}
          <div className="flex-1 min-w-0">
            <ChatDisplay
              title="Normal Response"
              content={result?.normal_response}
              isLoading={isLoading}
            />
          </div>

          {/* Steered Response */}
          <div className="flex-1 min-w-0">
            <ChatDisplay
              title="Steered Response"
              content={result?.steered_response}
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
