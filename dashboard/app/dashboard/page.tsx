"use client";

import { UserSession } from "@/components/UserSession";
import { authClient } from "@/lib/auth-client";
import { useState, useEffect } from "react";

export default function DashboardPage() {
  const { data: session } = authClient.useSession();
  const [organizations, setOrganizations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadOrganizations = async () => {
      try {
        if (session?.user) {
          const data = await authClient.organization.listMembers();
          setOrganizations(data?.organizations || []);
        }
      } catch (error) {
        console.error("Failed to load organizations:", error);
      } finally {
        setLoading(false);
      }
    };

    loadOrganizations();
  }, [session]);

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <UserSession />
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="border-4 border-dashed border-gray-200 rounded-lg h-96 p-6">
            {loading ? (
              <div className="text-center text-gray-600">
                Loading organizations...
              </div>
            ) : organizations.length === 0 ? (
              <div className="text-center">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No organizations yet
                </h3>
                <p className="text-gray-600 mb-4">
                  Create your first organization to get started
                </p>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  Create Organization
                </button>
              </div>
            ) : (
              <div>
                <h2 className="text-xl font-semibold mb-4">Your Organizations</h2>
                <ul className="space-y-2">
                  {organizations.map((org: any) => (
                    <li
                      key={org.id}
                      className="p-4 bg-white rounded border border-gray-200"
                    >
                      {org.name}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
