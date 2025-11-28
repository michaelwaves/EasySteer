import { getUser } from "@/lib/auth-server";

/**
 * Server Component that fetches and displays user profile
 * No client-side hydration needed for this data
 */
export async function UserProfile() {
  const user = await getUser();

  if (!user) {
    return <div>Not authenticated</div>;
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-4">User Profile</h2>
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium text-gray-600">Name</label>
          <p className="text-lg text-gray-900">{user.name}</p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-600">Email</label>
          <p className="text-lg text-gray-900">{user.email}</p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-600">User ID</label>
          <p className="text-lg font-mono text-gray-900">{user.id}</p>
        </div>
        {user.image && (
          <div>
            <label className="text-sm font-medium text-gray-600">Avatar</label>
            <img
              src={user.image}
              alt={user.name}
              className="w-20 h-20 rounded-full mt-2"
            />
          </div>
        )}
      </div>
    </div>
  );
}
