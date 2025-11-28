"use server";

import { getUser, requireAuth } from "@/lib/auth-server";

/**
 * Example Server Action that requires authentication
 * Can be called from Client Components without exposing auth logic
 */
export async function fetchUserData() {
  try {
    const user = await requireAuth();

    // Here you can:
    // - Query your database directly
    // - Call external APIs securely
    // - Access environment variables
    // - Keep secrets safe

    return {
      success: true,
      data: {
        userId: user.id,
        email: user.email,
        name: user.name,
      },
    };
  } catch (error) {
    console.error("Action failed:", error);
    return {
      success: false,
      error: "Unauthorized",
    };
  }
}

/**
 * Example Server Action that performs a protected operation
 */
export async function updateUserProfile(formData: {
  name: string;
  email: string;
}) {
  try {
    const user = await requireAuth();

    // Validate input
    if (!formData.name || !formData.email) {
      return {
        success: false,
        error: "Name and email are required",
      };
    }

    // Here you would:
    // - Update the database
    // - Send emails
    // - Log actions
    // - Trigger webhooks

    console.log(`User ${user.id} updating profile`, formData);

    return {
      success: true,
      message: "Profile updated successfully",
    };
  } catch (error) {
    console.error("Failed to update profile:", error);
    return {
      success: false,
      error: "Failed to update profile",
    };
  }
}

/**
 * Example Server Action for creating an organization
 */
export async function createOrganization(name: string, slug: string) {
  try {
    const user = await requireAuth();

    // Validate
    if (!name || !slug) {
      return {
        success: false,
        error: "Name and slug are required",
      };
    }

    // You can now:
    // - Call Better Auth API from server
    // - Create database records
    // - Send notifications
    // - Keep operation secure

    console.log(`User ${user.id} creating organization: ${name}`);

    return {
      success: true,
      data: {
        organizationId: "org_" + Math.random().toString(36).slice(2),
        name,
        slug,
        createdBy: user.id,
      },
    };
  } catch (error) {
    console.error("Failed to create organization:", error);
    return {
      success: false,
      error: "Failed to create organization",
    };
  }
}

/**
 * Example Server Action that checks user permissions
 */
export async function checkOrgAccess(_organizationId: string) {
  try {
    await requireAuth();

    // Here you would:
    // - Check database for user's org membership
    // - Verify role/permissions
    // - Return access info

    // Example check (implement with your database)
    const hasAccess = true; // Replace with actual DB check

    if (!hasAccess) {
      return {
        success: false,
        error: "You don't have access to this organization",
      };
    }

    return {
      success: true,
      canEdit: true,
      canDelete: false,
      canInvite: true,
    };
  } catch (error) {
    console.error("Failed to check access:", error);
    return {
      success: false,
      error: "Failed to check access",
    };
  }
}

/**
 * Example Server Action that logs an action
 */
export async function logAction(action: string, details?: any) {
  try {
    const user = await getUser();
    if (!user) return;

    // Log to database
    console.log(`[AUDIT LOG] User: ${user.id}, Action: ${action}`, details);

    // You could store this in a database table
    // await db.auditLog.create({
    //   userId: user.id,
    //   action,
    //   details,
    //   timestamp: new Date(),
    // });
  } catch (error) {
    console.error("Failed to log action:", error);
  }
}
