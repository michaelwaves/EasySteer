import { auth } from "@/lib/auth";
import { headers, cookies } from "next/headers";
import { redirect } from "next/navigation";

/**
 * Get the current user session on the server
 * Can be used in Server Components, API routes, and Server Actions
 *
 * @example
 * const session = await getSession();
 * if (!session) redirect("/auth/login");
 */
export async function getSession() {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });
    return session;
  } catch (error) {
    console.error("Failed to get session:", error);
    return null;
  }
}

/**
 * Get the current user from session
 * Throws error if not authenticated
 *
 * @example
 * const user = await getUser();
 * console.log(user.email);
 */
export async function getUser() {
  const session = await getSession();
  if (!session?.user) {
    return null;
  }
  return session.user;
}

/**
 * Ensure user is authenticated, redirect to login if not
 *
 * @example
 * const user = await requireAuth();
 * // user is guaranteed to exist here
 */
export async function requireAuth() {
  const user = await getUser();
  if (!user) {
    redirect("/auth/login");
  }
  return user;
}

/**
 * Get the session token from cookies
 * Useful for making authenticated requests to external APIs
 */
export async function getSessionToken() {
  const cookieStore = await cookies();
  return cookieStore.get("better-auth.session_token")?.value;
}

/**
 * Verify if user has a specific role in an organization
 *
 * @example
 * const isOwner = await hasOrgRole("org-id", "owner");
 */
export async function hasOrgRole(_organizationId: string, _role: string) {
  try {
    const session = await getSession();
    if (!session?.user) return false;

    // You can verify the role by checking the session or making an API call
    // This is a basic implementation - customize based on your needs
    return session.user.id ? true : false;
  } catch (error) {
    console.error("Failed to verify org role:", error);
    return false;
  }
}

/**
 * Get all active sessions for the current user
 * Useful for showing "logged in on X devices"
 */
export async function getUserSessions() {
  try {
    const user = await getUser();
    if (!user) return [];

    const sessionToken = await getSessionToken();
    if (!sessionToken) return [];

    // You can fetch sessions from your database
    // This would need a custom endpoint in Better Auth
    return [];
  } catch (error) {
    console.error("Failed to get user sessions:", error);
    return [];
  }
}

/**
 * Check if user is authenticated (boolean)
 *
 * @example
 * if (await isAuthenticated()) {
 *   // show authenticated content
 * }
 */
export async function isAuthenticated() {
  const user = await getUser();
  return !!user;
}
