import { getSession } from "@/lib/auth-server";
import { NextResponse } from "next/server";

/**
 * GET /api/data/user
 * Returns the current authenticated user data
 *
 * Requires authentication
 */
export async function GET() {
  try {
    const session = await getSession();

    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    // Return user data
    return NextResponse.json(
      {
        success: true,
        user: session.user,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching user:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
