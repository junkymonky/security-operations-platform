// Upserts MITRE ATT&CK reference data (official technique names and tactics) into
// mitre_techniques. Idempotent; safe to re-run.   pnpm db:seed-mitre
import { neon } from '@neondatabase/serverless'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')
let url = process.env.DATABASE_URL
if (!url) {
  const env = path.join(root, '.env.local')
  const m = fs.existsSync(env) && fs.readFileSync(env, 'utf8').match(/^DATABASE_URL=(.*)$/m)
  url = m && m[1].trim().replace(/^["']|["']$/g, '')
}
if (!url) { console.error('DATABASE_URL is not set'); process.exit(1) }

// Source: MITRE ATT&CK Enterprise matrix (attack.mitre.org).
const TECHNIQUES = [
  ['T1003', 'OS Credential Dumping', 'Credential Access', 'Adversaries may dump credentials from the operating system and software.'],
  ['T1021', 'Remote Services', 'Lateral Movement', 'Adversaries may use valid accounts to log into remote services.'],
  ['T1046', 'Network Service Discovery', 'Discovery', 'Adversaries may enumerate services running on remote hosts.'],
  ['T1048', 'Exfiltration Over Alternative Protocol', 'Exfiltration', 'Adversaries may steal data over a protocol other than the existing C2 channel.'],
  ['T1059', 'Command and Scripting Interpreter', 'Execution', 'Adversaries may abuse command and script interpreters to execute commands.'],
  ['T1059.001', 'Command and Scripting Interpreter: PowerShell', 'Execution', 'Adversaries may abuse PowerShell commands and scripts for execution.'],
  ['T1059.003', 'Command and Scripting Interpreter: Windows Command Shell', 'Execution', 'Adversaries may abuse the Windows command shell for execution.'],
  ['T1071', 'Application Layer Protocol', 'Command and Control', 'Adversaries may communicate using application layer protocols to avoid detection.'],
  ['T1071.001', 'Application Layer Protocol: Web Protocols', 'Command and Control', 'Adversaries may communicate using web protocols (HTTP/S) to blend in.'],
  ['T1078', 'Valid Accounts', 'Initial Access, Persistence, Privilege Escalation, Defense Evasion', 'Adversaries may obtain and abuse credentials of existing accounts.'],
  ['T1105', 'Ingress Tool Transfer', 'Command and Control', 'Adversaries may transfer tools or files from an external system.'],
  ['T1110', 'Brute Force', 'Credential Access', 'Adversaries may use brute force techniques to gain access to accounts.'],
  ['T1110.003', 'Brute Force: Password Spraying', 'Credential Access', 'Adversaries may use a single or small list of common passwords against many accounts.'],
  ['T1190', 'Exploit Public-Facing Application', 'Initial Access', 'Adversaries may exploit weaknesses in internet-facing software.'],
  ['T1486', 'Data Encrypted for Impact', 'Impact', 'Adversaries may encrypt data on target systems to interrupt availability.'],
  ['T1566', 'Phishing', 'Initial Access', 'Adversaries may send phishing messages to gain access to victim systems.'],
]

const sql = neon(url)
for (const [id, name, tactic, description] of TECHNIQUES) {
  await sql`INSERT INTO mitre_techniques (id, name, tactic, description) VALUES (${id}, ${name}, ${tactic}, ${description})
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, tactic = EXCLUDED.tactic, description = EXCLUDED.description`
}
const [{ n }] = await sql`SELECT count(*)::int AS n FROM mitre_techniques`
const missing = await sql`SELECT DISTINCT mitre_technique AS id FROM events
                          WHERE mitre_technique IS NOT NULL AND mitre_technique NOT IN (SELECT id FROM mitre_techniques)`
console.log(`mitre_techniques: ${n} rows.${missing.length ? ' Techniques in events without reference data: ' + missing.map((m) => m.id).join(', ') : ' All techniques referenced by events are covered.'}`)
