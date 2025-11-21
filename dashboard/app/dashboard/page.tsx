"use client";

import { UserSession } from "@/components/UserSession";
import { authClient } from "@/lib/auth-client";

export default function DashboardPage() {
  const { data: session, isPending } = authClient.useSession();

  if (isPending) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-600">Loading...</div>
      </div>
    );
  }

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
          <div className="border-4 border-dashed border-gray-200 rounded-lg p-6">
            <h2 className="text-2xl font-bold mb-6">Welcome to Your Dashboard</h2>

            {session?.user && (
              <div className="bg-white rounded-lg shadow p-6 mb-6">
                <h3 className="text-xl font-semibold mb-4">Your Profile</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Name:</span> {session.user.name}</p>
                  <p><span className="font-medium">Email:</span> {session.user.email}</p>
                  <p><span className="font-medium">User ID:</span> {session.user.id}</p>
                  {session.user.image && (
                    <img
                      src={session.user.image}
                      alt={session.user.name}
                      className="w-16 h-16 rounded-full mt-4"
                    />
                  )}
                </div>
              </div>
            )}

            <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
              <h3 className="text-lg font-semibold mb-2 text-blue-900">Getting Started</h3>
              <ul className="space-y-2 text-blue-900">
                <li>✓ Authentication is working!</li>
                <li>✓ You can create organizations to manage teams</li>
                <li>✓ Each organization can have multiple teams</li>
                <li>✓ Assign roles and permissions to members</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
