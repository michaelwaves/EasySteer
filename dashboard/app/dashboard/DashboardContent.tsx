"use client";

import { useState } from "react";
import { fetchUserData, createOrganization } from "./actions";

export function DashboardContent() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchUserData();
      if (result.success) {
        setData(result.data);
      } else {
        setError(result.error || "Failed to fetch data");
      }
    } catch (err) {
      setError("Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateOrg = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await createOrganization("My Organization", "my-org");
      if (result.success) {
        alert(`Organization created: ${result.data?.name}`);
      } else {
        setError(result.error || "Failed to create organization");
      }
    } catch (err) {
      setError("Failed to create organization");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>

      <div className="space-y-6">
        {/* Server-Side Data Fetching Example */}
        <div className="border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Server-Side Data Fetching</h3>
          <p className="text-gray-600 mb-4">
            Click the button below to fetch authenticated data from the server using a Server Action.
          </p>
          <button
            onClick={handleFetchData}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? "Fetching..." : "Fetch User Data"}
          </button>

          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}

          {data && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded">
              <pre className="text-sm overflow-auto">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Organization Creation Example */}
        <div className="border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Create Organization</h3>
          <p className="text-gray-600 mb-4">
            Use a Server Action to safely create organizations with server-side validation.
          </p>
          <button
            onClick={handleCreateOrg}
            disabled={loading}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400"
          >
            {loading ? "Creating..." : "Create Organization"}
          </button>
        </div>

        {/* Information Box */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-semibold text-blue-900 mb-2">How Server-Side Auth Works</h4>
          <ul className="text-blue-900 space-y-2 text-sm">
            <li>✓ Server Actions use <code className="bg-blue-100 px-1 rounded">requireAuth()</code> for protection</li>
            <li>✓ Client never sees authentication logic</li>
            <li>✓ Database queries happen securely on the server</li>
            <li>✓ Environment variables and secrets stay on the server</li>
            <li>✓ Better performance with reduced client-side code</li>
          </ul>
        </div>

        {/* Code Example */}
        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-auto">
          <p className="text-sm text-gray-400 mb-2">Server Action Example:</p>
          <pre className="text-xs">
{`"use server";

export async function fetchUserData() {
  const user = await requireAuth(); // Secure!

  // Safe to query database directly
  // Safe to use environment variables
  // Safe to call external APIs

  return { success: true, data: user };
}`}
          </pre>
        </div>
      </div>
    </div>
  );
}
