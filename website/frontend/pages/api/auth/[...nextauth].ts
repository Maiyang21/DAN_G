import NextAuth from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

export default NextAuth({
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        username: { label: 'Username', type: 'text' },
        password: { label: 'Password', type: 'password' }
      },
      async authorize(credentials) {
        if (!credentials?.username || !credentials?.password) {
          return null
        }

        // For demo purposes, use hardcoded credentials
        // In production, this would call your Railway backend
        if (credentials.username === 'admin' && credentials.password === 'admin123') {
          return {
            id: '1',
            name: 'Admin User',
            email: 'admin@dan-g-platform.com'
          }
        }

        return null
      }
    })
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = 'admin' // Set role directly
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        (session.user as any).role = token.role
        (session as any).accessToken = token.accessToken
      }
      return session
    }
  },
  pages: {
    signIn: '/',
    error: '/',
  },
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET,
})
