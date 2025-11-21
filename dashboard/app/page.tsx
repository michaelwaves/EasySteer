import { UserSession } from "@/components/UserSession";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">EasySteer</h1>
          <UserSession />
        </div>
      </nav>

      <main className="flex-1 max-w-7xl mx-auto w-full py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to EasySteer Dashboard
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Manage your teams and organizations with ease
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
            <div className="bg-white p-8 rounded-lg shadow">
              <div className="text-4xl mb-4">🔐</div>
              <h3 className="text-xl font-semibold mb-2">Secure Authentication</h3>
              <p className="text-gray-600">
                Sign in with email/password, GitHub, or Google
              </p>
            </div>

            <div className="bg-white p-8 rounded-lg shadow">
              <div className="text-4xl mb-4">👥</div>
              <h3 className="text-xl font-semibold mb-2">Organization Management</h3>
              <p className="text-gray-600">
                Create and manage organizations with team members
              </p>
            </div>

            <div className="bg-white p-8 rounded-lg shadow">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold mb-2">Team Collaboration</h3>
              <p className="text-gray-600">
                Organize teams within your organization
              </p>
            </div>
          </div>

          <div className="mt-12">
            <a
              href="/dashboard"
              className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
            >
              Go to Dashboard
            </a>
          </div>
        </div>
      </main>

      <footer className="bg-gray-100 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-gray-600">
          <p>&copy; 2024 EasySteer. Built with Next.js and Better Auth.</p>
        </div>
      </footer>
    </div>
  );
}
