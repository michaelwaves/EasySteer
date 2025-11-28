# Client-Side vs Server-Side Authentication

## Overview

Your dashboard now supports **both** client-side and server-side authentication. Here's when to use each.

## Quick Comparison

| Aspect | Client-Side | Server-Side |
|--------|------------|------------|
| **Best For** | UI state, real-time updates | Data fetching, security |
| **Where Used** | Client Components, Browser | Server Components, API Routes |
| **Database Access** | ❌ No (via API) | ✅ Yes (direct) |
| **Secrets Safe** | ❌ No (exposed to browser) | ✅ Yes (hidden) |
| **Performance** | Slower (API call) | Faster (direct access) |
| **Validation** | Client + Server | Server only |
| **SEO** | ❌ Client-rendered | ✅ Server-rendered |
| **User Feedback** | Instant, reactive | Requires form submission |

## Client-Side Authentication

### What It Is

Client-side authentication uses the `authClient` to manage authentication state in the browser.

### Best For

- Checking if user is logged in
- Showing/hiding UI based on auth state
- Sign in/sign up forms
- Real-time user feedback
- User session dropdown
- "Sign Out" button

### Example Usage

```typescript
// components/UserSession.tsx
"use client";

import { authClient } from "@/lib/auth-client";

export function UserSession() {
  const { data: session } = authClient.useSession();

  if (!session) {
    return <a href="/auth/login">Sign In</a>;
  }

  return <div>{session.user.name}</div>;
}
```

### Sign In Form

```typescript
// components/auth/SignInForm.tsx
"use client";

import { authClient } from "@/lib/auth-client";

export function SignInForm() {
  async function handleSignIn(email: string, password: string) {
    const response = await authClient.signIn.email({
      email,
      password,
    });

    if (!response.error) {
      // User signed in successfully
      // Session automatically updated
    }
  }

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      // Handle form...
    }}>
      {/* Form fields */}
    </form>
  );
}
```

### Where to Use

- Authentication UI components
- Login/Sign up pages
- User profile dropdown
- Session state hooks
- OAuth buttons

### When NOT to Use

- Fetching data from your database
- Calling external APIs securely
- Validation before data operations
- Storing sensitive information
- Checking permissions for data access

## Server-Side Authentication

### What It Is

Server-side authentication uses the `auth` instance and utility functions to verify authentication on the server.

### Best For

- Fetching user data from database
- Checking permissions before operations
- Securely calling external APIs
- Validating user input
- Logging actions for audit trail
- Creating/updating data

### Example Usage: Server Component

```typescript
// app/dashboard/UserProfile.tsx
import { getUser } from "@/lib/auth-server";

export async function UserProfile() {
  const user = await getUser();

  if (!user) {
    return <div>Not authenticated</div>;
  }

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

### Example Usage: Server Action

```typescript
// app/dashboard/actions.ts
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function updateProfile(formData: FormData) {
  const user = await requireAuth();
  const name = formData.get("name") as string;

  // Safe database operation
  await db.user.update(user.id, { name });

  return { success: true };
}
```

### Example Usage: API Route

```typescript
// app/api/data/user/route.ts
import { getSession } from "@/lib/auth-server";

export async function GET() {
  const session = await getSession();

  if (!session?.user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Fetch user data
  const userData = await db.user.findOne(session.user.id);

  return Response.json(userData);
}
```

### Where to Use

- Server Components (fetch data on render)
- Server Actions (form submissions)
- API Routes (external API calls)
- Protected data operations
- Permission checking
- Audit logging

### When NOT to Use

- Checking UI state
- Real-time user feedback
- Authentication forms (needs client-side interaction)
- Reactive UI updates
- Client-side validation feedback

## Practical Examples

### Example 1: Display User Profile

```typescript
// Best: Server Component (fastest, no client JS)

// app/dashboard/page.tsx
import { getUser } from "@/lib/auth-server";

export default async function DashboardPage() {
  const user = await getUser();

  if (!user) return <div>Not authenticated</div>;

  return (
    <div>
      <h1>Hello, {user.name}</h1>
      <p>Email: {user.email}</p>
    </div>
  );
}
```

### Example 2: Sign In Form

```typescript
// Best: Client Component (needs interactivity)

// components/auth/SignInForm.tsx
"use client";

import { authClient } from "@/lib/auth-client";
import { useState } from "react";

export function SignInForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const result = await authClient.signIn.email({
      email,
      password,
    });

    if (!result.error) {
      // Redirect or show success
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        type="email"
      />
      <input
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        type="password"
      />
      <button>Sign In</button>
    </form>
  );
}
```

### Example 3: Update User Data

```typescript
// Best: Server Action (secure, validated)

// app/dashboard/actions.ts
"use server";

import { requireAuth } from "@/lib/auth-server";

export async function updateUserName(name: string) {
  const user = await requireAuth();

  // Validate
  if (!name.trim()) {
    return { error: "Name cannot be empty" };
  }

  // Update database
  await db.user.update(user.id, { name });

  // Revalidate cache
  revalidatePath("/dashboard");

  return { success: true };
}
```

```typescript
// app/dashboard/EditProfile.tsx
"use client";

import { updateUserName } from "./actions";

export function EditProfile() {
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const name = new FormData(form).get("name");

    const result = await updateUserName(name as string);

    if (result.success) {
      alert("Profile updated!");
    } else {
      alert(result.error);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" required />
      <button type="submit">Update</button>
    </form>
  );
}
```

### Example 4: Fetch User Organizations

```typescript
// Best: Server Component with Server Action

// app/dashboard/Orgs.tsx
import { getUserOrgs } from "./actions";

export async function UserOrganizations() {
  // Server Component - fetches data on render
  const orgs = await getUserOrgs();

  return (
    <div>
      <h2>Your Organizations</h2>
      <ul>
        {orgs.map((org) => (
          <li key={org.id}>{org.name}</li>
        ))}
      </ul>
      <CreateOrgForm />
    </div>
  );
}

// Client Component - handles creation
"use client";

import { createOrg } from "./actions";

function CreateOrgForm() {
  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const data = new FormData(form);

    const result = await createOrg(
      data.get("name") as string,
      data.get("slug") as string
    );

    if (result.success) {
      // Revalidate and show new org
    }
  }

  return <form onSubmit={handleCreate}>...</form>;
}
```

### Example 5: Check Permissions

```typescript
// Best: Server-Side Check

// app/api/org/[id]/members/route.ts
import { getUser } from "@/lib/auth-server";

export async function GET(req: Request, { params }: { params: { id: string } }) {
  const user = await getUser();

  if (!user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Check if user is member
  const isMember = await db.organization.isMember(params.id, user.id);
  if (!isMember) {
    return Response.json({ error: "Forbidden" }, { status: 403 });
  }

  // Get org members
  const members = await db.organization.getMembers(params.id);

  return Response.json(members);
}
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Next.js App                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────┐   ┌──────────────────────┐
│  │ CLIENT COMPONENTS    │   │ SERVER COMPONENTS    │
│  ├──────────────────────┤   ├──────────────────────┤
│  │                      │   │                      │
│  │ • Sign In Form       │   │ • User Profile       │
│  │ • Sign Up Form       │   │ • Org List           │
│  │ • Dropdowns          │   │ • Settings           │
│  │ • Modals             │   │ • Static Content     │
│  │ • Real-time UI       │   │                      │
│  │                      │   │ Uses getUser()       │
│  │ Uses authClient      │   │ Direct DB access     │
│  │                      │   │                      │
│  └──────┬───────────────┘   └──────┬───────────────┘
│         │                          │
│         ▼                          ▼
│  ┌──────────────────────┐   ┌──────────────────────┐
│  │  CLIENT-SIDE AUTH    │   │  SERVER-SIDE AUTH    │
│  ├──────────────────────┤   ├──────────────────────┤
│  │                      │   │                      │
│  │ • authClient         │   │ • getUser()          │
│  │ • useSession()       │   │ • getSession()       │
│  │ • signIn.email()     │   │ • requireAuth()      │
│  │ • signIn.social()    │   │ • Server Actions     │
│  │ • signOut()          │   │ • API Routes         │
│  │                      │   │                      │
│  │ Browser State        │   │ Server-Side State    │
│  │                      │   │                      │
│  └──────────────────────┘   └──────────────────────┘
│
├─────────────────────────────────────────────────────┤
│              Better Auth Backend                     │
│  - Session Management                               │
│  - OAuth Integration                                │
│  - Database Persistence                             │
└─────────────────────────────────────────────────────┘
```

## Decision Tree

```
Need to...

├─ Check if user is logged in?
│  └─ Use Client: authClient.useSession()
│
├─ Show/hide UI based on auth?
│  └─ Use Client: const { data: session } = authClient.useSession()
│
├─ Fetch data from database?
│  └─ Use Server: const user = await getUser()
│
├─ Update/create data?
│  └─ Use Server Action: "use server"; export async function action()
│
├─ Call external API securely?
│  └─ Use Server Action or API Route
│
├─ Check permissions for data?
│  └─ Use Server: Check DB before returning data
│
├─ Need real-time reactivity?
│  └─ Use Client with Server Action
│
└─ Just rendering static content?
   └─ Use Server Component (fastest!)
```

## Best Practices

### ✅ DO:

1. **Use Server Components for data fetching**
   ```typescript
   const user = await getUser();
   ```

2. **Use Server Actions for mutations**
   ```typescript
   "use server";
   export async function updateProfile(data) { ... }
   ```

3. **Use Client Components for forms and interactivity**
   ```typescript
   "use client";
   <form action={updateProfile}>
   ```

4. **Check permissions on the server**
   ```typescript
   if (!await canAccess(user.id, resource)) {
     return { error: "Forbidden" };
   }
   ```

### ❌ DON'T:

1. **Don't fetch data on the client if server can do it**
   ```typescript
   // ❌ Bad
   useEffect(() => { fetch("/api/user") }, [])

   // ✅ Good
   const user = await getUser()
   ```

2. **Don't trust client-side auth checks alone**
   ```typescript
   // ❌ Bad
   if (session) { delete database }

   // ✅ Good
   const user = await requireAuth()
   if (!canDelete(user, item)) return
   ```

3. **Don't expose secrets to client**
   ```typescript
   // ❌ Bad
   const API_KEY = process.env.NEXT_PUBLIC_API_KEY

   // ✅ Good
   // Use secrets on server, client calls server endpoint
   ```

4. **Don't skip server-side validation**
   ```typescript
   // ❌ Bad
   if (clientSideValidation) { saveToDB() }

   // ✅ Good
   // Validate on server before saving
   ```

## Summary

| Use Case | Technology |
|----------|-----------|
| Auth UI State | Client + authClient |
| Sign In/Up Forms | Client + authClient |
| Check if logged in | Client + useSession() |
| Fetch Data | Server Component + getUser() |
| Update/Create | Server Action + requireAuth() |
| External APIs | Server Action + getSessionToken() |
| Protect Routes | Middleware + getSession() |
| API Endpoints | API Route + getSession() |

Use both together for the best user experience and security!

---

See **`SERVER_AUTH_GUIDE.md`** for detailed patterns and examples.
