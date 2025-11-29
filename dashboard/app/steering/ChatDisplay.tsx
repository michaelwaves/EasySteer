'use client';

interface ChatDisplayProps {
  title: string;
  content?: string;
  isLoading: boolean;
}

export function ChatDisplay({ title, content, isLoading }: ChatDisplayProps) {
  return (
    <div className="flex flex-col h-full bg-white border border-gray-200 rounded overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="space-y-3 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-5/6" />
            <div className="h-4 bg-gray-200 rounded w-4/6" />
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-3/6" />
          </div>
        ) : content ? (
          <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
            {content}
          </div>
        ) : (
          <div className="text-sm text-gray-400">
            Generate a response to see output here
          </div>
        )}
      </div>
    </div>
  );
}
