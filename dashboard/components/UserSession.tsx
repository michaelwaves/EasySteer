"use client";

import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

export function UserSession() {
  const router = useRouter();
  const { data: session, isPending } = authClient.useSession();

  if (isPending) {
    return <div className="text-gray-600">Loading...</div>;
  }

  if (!session) {
    return (
      <div className="flex gap-2">
        <a
          href="/auth/login"
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Sign In
        </a>
        <a
          href="/auth/signup"
          className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50"
        >
          Sign Up
        </a>
      </div>
    );
  }

  const handleSignOut = async () => {
    await authClient.signOut();
    router.push("/");
  };

  return (
    <div className="flex items-center gap-4">
      <div className="text-sm">
        <p className="font-medium">{session.user.name}</p>
        <p className="text-gray-600">{session.user.email}</p>
      </div>
      <button
        onClick={handleSignOut}
        className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
      >
        Sign Out
      </button>
    </div>
  );
}
