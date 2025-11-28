# Server-Side Authentication Guide

This guide covers how to use Better Auth for server-side authentication, including Server Components, Server Actions, and API routes.

## Overview

Server-side authentication in Next.js provides:

- **Security**: Authentication logic and secrets stay on the server
- **Performance**: Less JavaScript sent to the client
- **Simplicity**: Direct database access without API overhead
- **Trust**: Server-verified data before rendering to client

## Getting the Session on the Server

### In Server Components

```typescript
// app/dashboard/UserProfile.tsx
import { getUser } from "@/lib/auth-server";

export async function UserProfile() {
  const user = await getUser();

  if (!user) {
    return <div>Not authenticated</div>;
  }

  return <div>{user.email}</div>;
}
```

### In API Routes

```typescript
// app/api/data/user/route.ts
import { getSession } from "@/lib/auth-server";
import { NextResponse } from "next/server";

export async function GET() {
  const session = await getSession();

  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({ user: session.user });
}
```

### In Server Actions

```typescript
// app/dashboard/actions.ts
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function updateProfile(name: string) {
  const user = await requireAuth(); // Throws redirect if not auth'd

  // User is guaranteed to exist here
  console.log(`Updating profile for user ${user.id}`);

  return { success: true };
}
```

## Auth Server Utilities

The `lib/auth-server.ts` file provides these utilities:

### `getSession()`

Get the current session object (user + session data).

```typescript
const session = await getSession();
if (session?.user) {
  console.log(session.user.email);
}
```

**Returns:**
```typescript
{
  user: {
    id: string;
    email: string;
    name: string;
    image?: string;
  };
  session: {
    id: string;
    userId: string;
    expiresAt: Date;
    // ... other session data
  };
} | null
```

### `getUser()`

Get just the user object (shorter than getSession).

```typescript
const user = await getUser();
if (!user) return <div>Not authenticated</div>;
```

**Returns:** User object or `null`

### `requireAuth()`

Get the user, but throw a redirect if not authenticated.

```typescript
export async function DashboardPage() {
  const user = await requireAuth(); // Redirects to /auth/login if not auth'd
  // User is guaranteed to exist here
}
```

**Returns:** User object (never null)
**Throws:** Redirect to `/auth/login` if not authenticated

### `isAuthenticated()`

Simple boolean check.

```typescript
if (await isAuthenticated()) {
  // Show authenticated content
}
```

### `getSessionToken()`

Get the session token from cookies (for external API calls).

```typescript
const token = await getSessionToken();
if (token) {
  // Use token to call external APIs
  const response = await fetch("https://api.example.com/data", {
    headers: { Authorization: `Bearer ${token}` },
  });
}
```

## Server Actions

Server Actions are the recommended way to call server-side code from Client Components.

### Basic Server Action

```typescript
// app/dashboard/actions.ts
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function saveData(data: string) {
  const user = await requireAuth();

  // Safe to:
  // - Query database directly
  // - Use environment variables
  // - Call external APIs
  // - Validate sensitive data

  console.log(`User ${user.id} saved data`);

  return { success: true, message: "Data saved!" };
}
```

### Calling from Client Component

```typescript
// app/dashboard/SaveForm.tsx
"use client";

import { saveData } from "./actions";

export function SaveForm() {
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const result = await saveData("my data");
    if (result.success) {
      alert(result.message);
    }
  }

  return <form onSubmit={handleSubmit}>...</form>;
}
```

### Server Action with Form Data

```typescript
// app/dashboard/actions.ts
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function updateProfile(formData: FormData) {
  const user = await requireAuth();

  const name = formData.get("name") as string;
  const email = formData.get("email") as string;

  // Validate
  if (!name || !email) {
    return { success: false, error: "Missing fields" };
  }

  // Update database
  // await db.user.update(user.id, { name, email });

  return { success: true };
}
```

### Using in a Form

```typescript
// app/dashboard/ProfileForm.tsx
"use client";

import { updateProfile } from "./actions";

export function ProfileForm() {
  return (
    <form action={updateProfile}>
      <input name="name" required />
      <input name="email" type="email" required />
      <button type="submit">Update</button>
    </form>
  );
}
```

## API Routes

For external API calls or more complex scenarios:

```typescript
// app/api/data/organizations/route.ts
import { getUser } from "@/lib/auth-server";
import { NextResponse } from "next/server";

export async function GET() {
  const user = await getUser();

  if (!user) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 }
    );
  }

  // Query organizations for this user
  // const orgs = await db.organization.findMany({
  //   where: { members: { some: { userId: user.id } } }
  // });

  return NextResponse.json({ organizations: [] });
}

export async function POST(request: Request) {
  const user = await getUser();

  if (!user) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 }
    );
  }

  const body = await request.json();

  // Validate
  if (!body.name) {
    return NextResponse.json(
      { error: "Name is required" },
      { status: 400 }
    );
  }

  // Create organization
  // const org = await db.organization.create({
  //   name: body.name,
  //   userId: user.id
  // });

  return NextResponse.json({ success: true, organization: {} });
}
```

## Pattern: Server Component with Server Action

This is the most common pattern:

```typescript
// app/dashboard/MyComponent.tsx
import { fetchData } from "./actions";

export async function MyComponent() {
  // Server Component - fetches data on render
  const data = await fetchData();

  return (
    <div>
      {/* Use data */}
      <UpdateForm data={data} />
    </div>
  );
}

// Client Component inside Server Component
"use client";

import { updateData } from "./actions";

function UpdateForm({ data }: { data: any }) {
  async function handleUpdate() {
    await updateData(data.id);
  }

  return <button onClick={handleUpdate}>Update</button>;
}
```

## Security Best Practices

### ✅ DO:

- Use `requireAuth()` to ensure authentication before operations
- Keep secrets and environment variables on the server
- Validate all input on the server
- Check permissions before returning data
- Log important actions for audit trails
- Return minimal data needed by client

```typescript
export async function getOrgData(orgId: string) {
  const user = await requireAuth();

  // Check if user has access to this org
  const isMember = await db.organization.isMember(orgId, user.id);
  if (!isMember) {
    return { error: "Not a member of this organization" };
  }

  // Return only necessary data
  const org = await db.organization.findById(orgId);
  return {
    id: org.id,
    name: org.name,
    // Don't return: billing info, API keys, etc.
  };
}
```

### ❌ DON'T:

- Trust client-side authentication state alone
- Expose sensitive data to the client
- Skip validation on the server
- Return more data than needed
- Use user-provided IDs in database queries without verification

```typescript
// ❌ Bad - trusts client
export async function deleteOrg(orgId: string) {
  await db.organization.delete(orgId); // No auth check!
}

// ✅ Good - verifies ownership
export async function deleteOrg(orgId: string) {
  const user = await requireAuth();
  const org = await db.organization.findById(orgId);

  if (org.userId !== user.id) {
    return { error: "Not authorized" };
  }

  await db.organization.delete(orgId);
}
```

## Database Access

When using Server Components or Server Actions, you can query your database directly:

```typescript
// app/dashboard/actions.ts
"use server";

import { db } from "@/lib/db"; // Your database client
import { requireAuth } from "@/lib/auth-server";

export async function getOrganizations() {
  const user = await requireAuth();

  // Direct database query
  const orgs = await db.organization.findMany({
    where: { members: { some: { userId: user.id } } },
    include: { members: { take: 5 } },
  });

  return { success: true, organizations: orgs };
}
```

## Working with Better Auth API

You can also call Better Auth methods from the server:

```typescript
import { auth } from "@/lib/auth";

// Create organization via Better Auth
export async function createOrganization(name: string, slug: string) {
  const user = await requireAuth();

  const org = await auth.api.createOrganization(
    {
      name,
      slug,
    },
    {
      headers: await headers(), // Pass request headers for context
    }
  );

  return org;
}
```

## Caching Server Components

Server Components can be cached for better performance:

```typescript
// Cache for 60 seconds
export const revalidate = 60;

export async function UserOrganizations() {
  const user = await getUser();
  const orgs = await db.organization.findMany({
    where: { members: { some: { userId: user?.id } } },
  });

  return <div>{/* Display orgs */}</div>;
}
```

You can also use `revalidatePath()` in Server Actions:

```typescript
"use server";

import { revalidatePath } from "next/cache";
import { requireAuth } from "@/lib/auth-server";

export async function createOrg(name: string) {
  const user = await requireAuth();

  // Create org...

  // Revalidate the page cache
  revalidatePath("/dashboard");

  return { success: true };
}
```

## Middleware vs Server-Side Auth

| Aspect | Middleware | Server-Side Auth |
|--------|-----------|-----------------|
| **When** | Before route handler | Inside route/component |
| **Use Case** | Redirect unauthenticated users | Get user data, check permissions |
| **Database** | Can't query database | Can query database |
| **Performance** | Early rejection | Full request processing |

Combine both:

```typescript
// middleware.ts - Protect the route
// Server Component - Check permissions and fetch data
// Server Action - Perform privileged operations
```

## Error Handling

```typescript
"use server";

export async function riskyOperation() {
  try {
    const user = await requireAuth(); // May redirect
    // Perform operation
    return { success: true };
  } catch (error) {
    if (error instanceof Error) {
      return { success: false, error: error.message };
    }
    return { success: false, error: "Unknown error" };
  }
}
```

## Testing Server Actions

```typescript
// __tests__/actions.test.ts
import { fetchUserData } from "@/app/dashboard/actions";

jest.mock("@/lib/auth-server", () => ({
  requireAuth: jest.fn().mockResolvedValue({
    id: "test-user",
    email: "test@example.com",
  }),
}));

test("fetchUserData returns user data", async () => {
  const result = await fetchUserData();
  expect(result.success).toBe(true);
  expect(result.data.email).toBe("test@example.com");
});
```

## Summary

Server-side authentication with Better Auth:

1. **Server Components**: Fetch data that requires auth
2. **Server Actions**: Mutations with auth verification
3. **API Routes**: External API endpoints with auth
4. **Middleware**: Early route protection

Use all three together for a complete, secure authentication system!

See the example files:
- `lib/auth-server.ts` - Auth utilities
- `app/dashboard/actions.ts` - Server Action examples
- `app/dashboard/UserProfile.tsx` - Server Component example
- `app/dashboard/DashboardContent.tsx` - Client Component with Server Actions
- `app/api/data/user/route.ts` - API route example
