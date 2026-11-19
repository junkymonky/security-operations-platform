// Analyst identity shown in the console. There is no auth/users table yet, so the
// values come from server configuration (.env.local): ANALYST_NAME, ANALYST_ROLE, ORG_NAME.
export interface AnalystProfile {
  name: string
  role: string
  org: string
}

export function analystProfile(): AnalystProfile {
  return {
    name: process.env.ANALYST_NAME || '',
    role: process.env.ANALYST_ROLE || 'Security analyst',
    org: process.env.ORG_NAME || '',
  }
}

export function initials(text: string, fallback = '?'): string {
  const parts = text.trim().split(/\s+/).filter(Boolean)
  if (!parts.length) return fallback
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase()
}
