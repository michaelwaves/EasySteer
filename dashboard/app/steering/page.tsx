'use client';

import { useState } from 'react';
import { SteeringComparison } from './SteeringComparison';

export default function SteeringPage() {
  return (
    <div className="min-h-screen bg-white">
      <nav className="border-b border-gray-200">
        <div className="max-w-full mx-auto px-6 py-4">
          <h1 className="text-2xl font-semibold font-sans text-gray-900">AI Feels 😃😭😡😨😮🤮</h1>
        </div>
      </nav>
      <main className="flex h-[calc(100vh-73px)]">
        <SteeringComparison />
      </main>
    </div>
  );
}
