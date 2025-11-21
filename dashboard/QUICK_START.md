# Quick Start Guide - Better Auth Implementation

## Project is Ready!

Your Next.js dashboard with Better Auth has been successfully set up. Here's what you need to do to get started:

## 1. Environment Variables

Create/update `.env.local` in the project root:

```env
# Database (Required)
DATABASE_URL=postgresql://user:password@localhost:5432/dashboard

# Better Auth (Required)
BETTER_AUTH_SECRET=your-secret-key-min-32-chars
BETTER_AUTH_URL=http://localhost:3000

# OAuth Providers (Optional but recommended)
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Public URL
NEXT_PUBLIC_BETTER_AUTH_URL=http://localhost:3000
```

## 2. Database Setup

### Using Docker:
```bash
docker run --name postgres-dashboard \
  -e POSTGRES_DB=dashboard \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d postgres:latest
```

### Or connect to existing PostgreSQL:
Just update `DATABASE_URL` in `.env.local`

## 3. OAuth Setup (Optional)

### GitHub OAuth:
1. Go to https://github.com/settings/developers
2. Click "New OAuth App"
3. Set Authorization callback URL to: `http://localhost:3000/api/auth/callback/github`
4. Copy Client ID and Secret to `.env.local`

### Google OAuth:
1. Go to https://console.cloud.google.com/
2. Create project > Go to Credentials
3. Create OAuth 2.0 Client ID (Web application)
4. Add redirect URI: `http://localhost:3000/api/auth/callback/google`
5. Copy Client ID and Secret to `.env.local`

## 4. Run Development Server

```bash
npm run dev
```

Visit: `http://localhost:3000`

## 5. Test Authentication

- Click "Sign In" or "Sign Up"
- Try email/password authentication
- Try GitHub/Google OAuth (if configured)
- Access `/dashboard` (protected route)

## Available Routes

### Public Routes:
- `/` - Home page
- `/auth/login` - Login page
- `/auth/signup` - Sign up page

### Protected Routes (require authentication):
- `/dashboard` - User dashboard
- `/settings` - Settings (add as needed)
- `/organizations` - Organizations management

### API Routes:
- `/api/auth/*` - All Better Auth endpoints

## Project Structure

```
dashboard/
├── lib/
│   ├── auth.ts              # Server auth config
│   └── auth-client.ts       # Client auth config
├── app/
│   ├── page.tsx             # Home
│   ├── api/auth/[...all]/   # Auth API
│   ├── auth/                # Auth pages
│   └── dashboard/           # Protected page
├── components/
│   ├── auth/                # Auth forms
│   └── UserSession.tsx      # User display
└── middleware.ts            # Route protection
```

## Authentication Methods

### 1. Email & Password
```typescript
// Sign up
await authClient.signUp.email({
  email: "user@example.com",
  password: "password123",
  name: "User Name"
});

// Sign in
await authClient.signIn.email({
  email: "user@example.com",
  password: "password123"
});
```

### 2. GitHub/Google OAuth
```typescript
await authClient.signIn.social({
  provider: "github" | "google"
});
```

### 3. Sign Out
```typescript
await authClient.signOut();
```

## Session Management

```typescript
import { authClient } from "@/lib/auth-client";

// Get current session (client)
const { data: session } = authClient.useSession();

// Get session on server
import { auth } from "@/lib/auth";
import { headers } from "next/headers";

const session = await auth.api.getSession({
  headers: await headers()
});
```

## Organization Features

```typescript
// Create organization
await authClient.organization.create({
  name: "My Company",
  slug: "my-company"
});

// List organizations
const orgs = await authClient.organization.list();

// Create team
await authClient.organization.createTeam({
  name: "Engineering",
  organizationId: "org-id"
});

// Add member
await authClient.organization.addMember({
  organizationId: "org-id",
  userId: "user-id",
  role: "member"
});
```

## Common Commands

```bash
# Development
npm run dev

# Build
npm run build

# Production start
npm start

# Lint
npm run lint

# Run migrations
npx @better-auth/cli migrate
```

## Key Files

| File | Purpose |
|------|---------|
| `lib/auth.ts` | Server configuration, providers, database |
| `lib/auth-client.ts` | Client configuration, client plugins |
| `app/api/auth/[...all]/route.ts` | API route handler |
| `middleware.ts` | Route protection middleware |
| `components/auth/*` | Sign in/up forms |
| `.env.local` | Environment variables |

## Features Implemented

✅ Email & Password authentication
✅ GitHub OAuth sign-in
✅ Google OAuth sign-in
✅ Organizations management
✅ Teams within organizations
✅ Role-based member management
✅ Protected routes with middleware
✅ User session management
✅ Secure database storage

## Production Deployment

1. **Generate BETTER_AUTH_SECRET**:
   ```bash
   openssl rand -base64 32
   ```

2. **Update env vars** on your hosting platform

3. **Update OAuth callbacks** to production URLs

4. **Test everything** before deploying

See `BETTER_AUTH_SETUP.md` for detailed deployment guide.

## Troubleshooting

### Database Connection Failed
- Check `DATABASE_URL` format
- Ensure PostgreSQL is running
- Verify network connectivity

### OAuth Redirect Error
- Verify callback URL matches exactly in provider settings
- Check client ID and secret
- Clear browser cookies

### Session Not Persisting
- Enable cookies in browser
- Check `BETTER_AUTH_URL` and `NEXT_PUBLIC_BETTER_AUTH_URL`
- Check browser console for errors

## Next Steps

1. ✅ Set up environment variables
2. ✅ Start development server
3. ✅ Test authentication flows
4. ✅ Customize UI (forms, pages, etc.)
5. ✅ Add email provider for notifications
6. ✅ Configure email verification
7. ✅ Implement additional features
8. ✅ Deploy to production

## Resources

- [Better Auth Docs](https://better-auth.com/docs)
- [Next.js Docs](https://nextjs.org/docs)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)

---

**Ready to build?** Start with: `npm run dev`
