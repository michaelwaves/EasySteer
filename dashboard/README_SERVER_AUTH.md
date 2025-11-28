# Server-Side Authentication Guide

Welcome! Your dashboard now has complete server-side authentication. Here's everything you need to know.

## Quick Links

- **Starting now?** Read: `CLIENT_VS_SERVER_AUTH.md`
- **Want examples?** See: `app/dashboard/actions.ts`
- **Need reference?** Check: `SERVER_AUTH_GUIDE.md`
- **Implementation details?** View: `FINAL_SERVER_AUTH_SUMMARY.md`

## Three Ways to Authenticate

### 1. Server Component (Data Fetching)

Fetch data that requires authentication:

```typescript
// app/dashboard/Profile.tsx
import { getUser } from "@/lib/auth-server";

export async function UserProfile() {
  const user = await getUser();
  return <div>Hello, {user.name}</div>;
}
```

**Use when:** Displaying authenticated data, static content, initial page load

### 2. Server Action (Mutations)

Safely update data with authentication:

```typescript
// app/dashboard/actions.ts
"use server";
import { requireAuth } from "@/lib/auth-server";

export async function updateProfile(formData: FormData) {
  const user = await requireAuth();
  const name = formData.get("name");

  // Update database
  await db.user.update(user.id, { name });

  // Revalidate cache
  revalidatePath("/profile");
}
```

Call from Client Component:

```typescript
"use client";
export function EditForm() {
  return (
    <form action={updateProfile}>
      <input name="name" />
      <button type="submit">Save</button>
    </form>
  );
}
```

**Use when:** Form submissions, creating/updating/deleting data

### 3. API Route (External Access)

Standard HTTP endpoints for external integrations:

```typescript
// app/api/data/user/route.ts
import { getSession } from "@/lib/auth-server";

export async function GET() {
  const session = await getSession();

  if (!session?.user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  return Response.json(session.user);
}
```

**Use when:** Mobile apps, third-party integrations, REST API

## Available Utilities

All in `lib/auth-server.ts`:

```typescript
import { getUser, getSession, requireAuth, isAuthenticated, getSessionToken } from "@/lib/auth-server";

// Get current user (returns user or null)
const user = await getUser();

// Get full session (user + session data)
const session = await getSession();

// Ensure authenticated (redirects if not)
const user = await requireAuth();

// Check if authenticated (boolean)
const loggedIn = await isAuthenticated();

// Get session token for external APIs
const token = await getSessionToken();
```

## Common Patterns

### Pattern 1: Protected Server Component

```typescript
import { getUser } from "@/lib/auth-server";

export async function ProtectedPage() {
  const user = await getUser();

  if (!user) {
    return <div>Please sign in</div>;
  }

  const data = await db.user.getData(user.id);

  return <div>{data}</div>;
}
```

### Pattern 2: Form with Server Action

```typescript
// Form component
"use client";

import { updateProfile } from "./actions";

export function ProfileForm() {
  return (
    <form action={updateProfile}>
      <input name="name" required />
      <input name="email" type="email" required />
      <button type="submit">Save</button>
    </form>
  );
}

// Server Action
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function updateProfile(formData: FormData) {
  const user = await requireAuth();

  const data = {
    name: formData.get("name"),
    email: formData.get("email"),
  };

  // Validate
  if (!data.name || !data.email) {
    return { error: "All fields required" };
  }

  // Update
  await db.user.update(user.id, data);

  // Revalidate
  revalidatePath("/profile");

  return { success: true };
}
```

### Pattern 3: Permission Checking

```typescript
"use server";

export async function deleteOrganization(orgId: string) {
  const user = await requireAuth();

  // Get organization
  const org = await db.organization.findOne(orgId);

  // Check permission
  if (org.userId !== user.id) {
    return { error: "Not authorized" };
  }

  // Delete
  await db.organization.delete(orgId);

  return { success: true };
}
```

### Pattern 4: API Endpoint with Auth

```typescript
// app/api/organizations/route.ts

import { getSession } from "@/lib/auth-server";

export async function GET() {
  const session = await getSession();

  if (!session?.user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Get user's organizations
  const orgs = await db.organization.findMany({
    where: { members: { some: { userId: session.user.id } } },
  });

  return Response.json({ organizations: orgs });
}

export async function POST(request: Request) {
  const session = await getSession();

  if (!session?.user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await request.json();

  // Create organization
  const org = await db.organization.create({
    name: body.name,
    userId: session.user.id,
  });

  return Response.json({ organization: org }, { status: 201 });
}
```

## Security Best Practices

✅ **Always check auth on server**
```typescript
const user = await requireAuth(); // Good!
```

❌ **Never skip verification**
```typescript
// Bad - trust user input!
await db.delete(requestBody.id);

// Good - verify ownership
const user = await requireAuth();
const item = await db.findOne(requestBody.id);
if (item.userId !== user.id) throw Error("Forbidden");
```

✅ **Validate input server-side**
```typescript
export async function createPost(title: string) {
  const user = await requireAuth();

  // Server-side validation
  if (!title || title.length < 3) {
    return { error: "Title must be at least 3 characters" };
  }

  // Safe to use
  await db.post.create({ title, userId: user.id });
}
```

✅ **Keep secrets on server**
```typescript
// ✅ Good - API key on server only
export async function callExternalAPI() {
  const apiKey = process.env.EXTERNAL_API_KEY;
  const response = await fetch("...", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  return response.json();
}

// ❌ Bad - API key exposed to browser
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
```

## Performance Tips

💡 **Use Server Components for static data**
```typescript
// Fast - no client JavaScript
export async function UserProfile() {
  const user = await getUser();
  return <div>{user.name}</div>;
}
```

💡 **Combine Server + Client for interactivity**
```typescript
// Server Component fetches initial data
export async function Dashboard() {
  const data = await fetchData();
  return <DashboardClient initialData={data} />;
}

// Client Component handles interactions
"use client";
function DashboardClient({ initialData }) {
  // Interactive features here
}
```

💡 **Revalidate after mutations**
```typescript
"use server";

export async function updateUser(data: any) {
  await db.user.update(data);
  revalidatePath("/profile"); // Fresh data next time
}
```

## Debugging

### Getting User Info

```typescript
const user = await getUser();
console.log(user); // { id, email, name, image }
```

### Checking Session

```typescript
const session = await getSession();
console.log(session); // { user, session }
```

### Handling Errors

```typescript
try {
  const user = await requireAuth();
  // Do something
} catch (error) {
  console.error("Auth failed:", error);
  // User was redirected to login
}
```

## Next Steps

1. **Build something!** Create Server Actions for your features
2. **Add database** Connect real database in Server Actions
3. **Implement permissions** Check access before operations
4. **Add logging** Track important actions
5. **Test thoroughly** Test all auth flows

## Examples in Your Code

Already set up for you:

- **Server Component:** `app/dashboard/UserProfile.tsx`
- **Server Actions:** `app/dashboard/actions.ts`
- **API Route:** `app/api/data/user/route.ts`
- **Client + Server:** `app/dashboard/DashboardContent.tsx`

Study these to understand the patterns!

## Common Issues

### "User is undefined"
Make sure you called `requireAuth()` not `getUser()`. `requireAuth()` redirects if not authenticated.

```typescript
// ✅ Correct - redirects if not auth'd
const user = await requireAuth();

// ❌ Wrong - might be null
const user = await getUser();
if (!user) return; // Manually handle
```

### "Session is null in API route"
Make sure headers are passed correctly:

```typescript
// ✅ Correct - get headers from request
export async function GET(request: Request) {
  const session = await getSession({
    headers: request.headers,
  });
}

// Better - use from `lib/auth-server`
import { getSession } from "@/lib/auth-server";
const session = await getSession();
```

### "Secrets visible in client"
Never use `NEXT_PUBLIC_` prefix for secrets:

```typescript
// ✅ Good - only on server
const DB_URL = process.env.DATABASE_URL;

// ❌ Bad - exposed to client!
const DB_URL = process.env.NEXT_PUBLIC_DATABASE_URL;
```

## Resources

- `CLIENT_VS_SERVER_AUTH.md` - Understand when to use each
- `SERVER_AUTH_GUIDE.md` - Comprehensive guide
- `FINAL_SERVER_AUTH_SUMMARY.md` - Implementation details
- `lib/auth-server.ts` - Available utilities
- `app/dashboard/actions.ts` - Server Action examples

## Support

Questions? Check:
1. The guides above
2. Code examples in `app/dashboard/`
3. Better Auth docs at https://better-auth.com

---

**Ready to build?** Run `npm run dev` and start creating Server Actions! 🚀
