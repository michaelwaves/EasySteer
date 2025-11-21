# Better Auth Setup Guide

This guide covers the complete setup of Better Auth with email+password, GitHub OAuth, Google OAuth, and organizations management in the EasySteer Dashboard.

## Prerequisites

- Node.js 18+
- PostgreSQL database
- GitHub OAuth credentials (optional)
- Google OAuth credentials (optional)

## Installation

All dependencies have been installed:
```bash
npm install better-auth pg
```

## Environment Variables

Create a `.env.local` file in the project root with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dashboard

# Better Auth
BETTER_AUTH_SECRET=your-secret-key-here-min-32-characters
BETTER_AUTH_URL=http://localhost:3000

# GitHub OAuth (optional)
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Next.js Public URLs
NEXT_PUBLIC_BETTER_AUTH_URL=http://localhost:3000
```

### Generating BETTER_AUTH_SECRET

Use OpenSSL to generate a secure secret:
```bash
openssl rand -base64 32
```

## Setting Up OAuth Providers

### GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in the application details:
   - **Application name**: EasySteer Dashboard
   - **Homepage URL**: `http://localhost:3000` (or your production URL)
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
4. Copy the Client ID and Client Secret to `.env.local`
5. **Important**: Make sure your GitHub app has the `user:email` scope enabled

### Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Go to APIs & Services > Credentials
4. Click "Create Credentials" > "OAuth 2.0 Client ID" > "Web application"
5. Add authorized redirect URIs:
   - `http://localhost:3000/api/auth/callback/google`
   - Your production URL (e.g., `https://yourdomain.com/api/auth/callback/google`)
6. Copy the Client ID and Client Secret to `.env.local`

## Database Setup

### Option 1: Using Docker

```bash
docker run --name postgres-dashboard \
  -e POSTGRES_DB=dashboard \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d postgres:latest
```

### Option 2: Local PostgreSQL

Ensure PostgreSQL is running and create a database:

```sql
CREATE DATABASE dashboard;
```

### Option 3: Hosted Database

Use a hosted PostgreSQL service like:
- Supabase
- Railway
- Vercel Postgres
- AWS RDS

## Running Database Migrations

Better Auth automatically creates required tables on first run. For explicit migration management, you can use the Better Auth CLI:

```bash
npx @better-auth/cli migrate
```

Or generate schema manually:

```bash
npx @better-auth/cli generate
```

## Project Structure

### Key Files

```
dashboard/
├── lib/
│   ├── auth.ts                 # Better Auth server configuration
│   └── auth-client.ts          # Better Auth client configuration
├── app/
│   ├── page.tsx                # Home page with user session display
│   ├── api/
│   │   └── auth/
│   │       └── [...all]/
│   │           └── route.ts    # Auth API routes handler
│   ├── auth/
│   │   ├── login/
│   │   │   └── page.tsx        # Login page
│   │   └── signup/
│   │       └── page.tsx        # Sign up page
│   └── dashboard/
│       └── page.tsx            # Protected dashboard page
├── components/
│   ├── auth/
│   │   ├── SignInForm.tsx      # Email/password & OAuth sign in
│   │   └── SignUpForm.tsx      # Email/password sign up
│   └── UserSession.tsx         # User session display component
├── middleware.ts               # Protected routes middleware
├── .env.local                  # Environment variables
└── BETTER_AUTH_SETUP.md        # This file
```

## Features

### Authentication Methods

1. **Email & Password**
   - Sign up with email and password (minimum 8 characters)
   - Sign in with email and password
   - Password validation and security

2. **GitHub OAuth**
   - One-click sign in with GitHub
   - Automatically retrieves user email and profile info
   - Links GitHub account to existing user

3. **Google OAuth**
   - One-click sign in with Google
   - Automatically retrieves user email and profile info
   - Links Google account to existing user

### Organizations

- Create organizations
- Manage organization members
- Assign roles to members
- Organize teams within organizations
- Invite members via email

### Teams

- Create teams within organizations
- Manage team members
- Organize work by team

## Usage

### Client-Side Session Management

```tsx
import { authClient } from "@/lib/auth-client";

export function MyComponent() {
  const { data: session, isPending } = authClient.useSession();

  if (isPending) return <div>Loading...</div>;
  if (!session) return <div>Not authenticated</div>;

  return <div>Welcome, {session.user.name}!</div>;
}
```

### Server-Side Session Management

```tsx
import { auth } from "@/lib/auth";
import { headers } from "next/headers";

export async function MyServerComponent() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) return <div>Not authenticated</div>;
  return <div>Welcome, {session.user.name}!</div>;
}
```

### Sign In

```tsx
await authClient.signIn.email({
  email: "user@example.com",
  password: "password123",
});

// Or with social providers
await authClient.signIn.social({
  provider: "github" | "google",
});
```

### Sign Up

```tsx
await authClient.signUp.email({
  email: "user@example.com",
  password: "password123",
  name: "User Name",
});
```

### Sign Out

```tsx
await authClient.signOut();
```

### Create Organization

```tsx
await authClient.organization.create({
  name: "My Organization",
  slug: "my-org",
});
```

### Manage Teams

```tsx
// Create team
await authClient.organization.createTeam({
  name: "My Team",
  organizationId: "org-id",
});

// List teams
await authClient.organization.listTeams({
  organizationId: "org-id",
});
```

## Protected Routes

The following routes are protected by middleware and require authentication:

- `/dashboard` - Main dashboard
- `/settings` - User settings
- `/organizations` - Organization management

Unauthenticated users are redirected to `/auth/login` with a callback URL.

## API Routes

All Better Auth endpoints are available at `/api/auth/*`:

- `POST /api/auth/sign-up/email` - Sign up with email
- `POST /api/auth/sign-in/email` - Sign in with email
- `POST /api/auth/sign-in/social` - Sign in with OAuth provider
- `POST /api/auth/sign-out` - Sign out
- `GET /api/auth/session` - Get current session
- `POST /api/auth/organization/create` - Create organization
- And many more...

See [Better Auth Documentation](https://better-auth.com/docs) for complete API reference.

## Session Management

- Sessions expire after 7 days by default
- Session activity is tracked and can be refreshed
- Multiple sessions per user are supported
- Sessions are stored in a secure database

## Security

- Passwords are hashed using bcrypt
- Session tokens are stored securely
- CSRF protection is enabled
- OAuth tokens are stored securely in the database
- Password minimum length: 8 characters

## Troubleshooting

### Database Connection Issues

Check your `DATABASE_URL` format:
```
postgresql://username:password@localhost:5432/database_name
```

### OAuth Callback Errors

1. Ensure callback URLs match exactly in provider settings and `.env.local`
2. Check that client ID and secret are correct
3. For GitHub, ensure `user:email` scope is enabled
4. For Google, check that authorized redirect URIs are set correctly

### Session Not Persisting

1. Check that cookies are enabled in your browser
2. Verify `BETTER_AUTH_URL` and `NEXT_PUBLIC_BETTER_AUTH_URL` are set correctly
3. Check browser console for CORS or cookie errors

### Email Provider Not Configured

Currently, email sign-in/sign-up works without a mail provider (local development). For production, configure an email provider in `lib/auth.ts`.

## Deployment

### Environment Variables

For production deployment (Vercel, Railway, etc.):

1. Set all environment variables in your deployment platform's settings
2. Update callback URLs to use your production domain
3. Update GitHub and Google OAuth callback URLs to your production domain
4. Use a strong `BETTER_AUTH_SECRET` (at least 32 characters)

### Database

Ensure your PostgreSQL database is:
- Accessible from your production environment
- Configured with proper backups
- Using SSL/TLS for connections

## Next Steps

1. Set up your environment variables in `.env.local`
2. Ensure PostgreSQL is running
3. Run the development server: `npm run dev`
4. Visit `http://localhost:3000`
5. Create an account or sign in with OAuth

## Resources

- [Better Auth Documentation](https://better-auth.com/docs)
- [Better Auth GitHub](https://github.com/better-auth/better-auth)
- [Next.js Documentation](https://nextjs.org/docs)

## Support

For issues with Better Auth, check:
- [Better Auth Issues](https://github.com/better-auth/better-auth/issues)
- [Better Auth Discord](https://discord.gg/better-auth)
