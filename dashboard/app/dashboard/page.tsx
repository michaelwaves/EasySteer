import { Suspense } from "react";
import { UserSession } from "@/components/UserSession";
import { UserProfile } from "./UserProfile";
import { DashboardContent } from "./DashboardContent";

export default function DashboardPage() {
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
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Server Component - User Profile */}
            <div className="lg:col-span-1">
              <Suspense fallback={<div className="bg-white rounded-lg shadow p-6 animate-pulse h-64" />}>
                <UserProfile />
              </Suspense>
            </div>

            {/* Client Component - Dashboard Content */}
            <div className="lg:col-span-2">
              <Suspense fallback={<div className="bg-white rounded-lg shadow p-6 animate-pulse h-96" />}>
                <DashboardContent />
              </Suspense>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
