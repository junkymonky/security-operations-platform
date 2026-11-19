// Cross-platform helper for the Python ML service.
//   node scripts/ml.mjs setup   create ml-service/.venv and install requirements
//   node scripts/ml.mjs dev     start FastAPI on http://127.0.0.1:8000 with reload
//   node scripts/ml.mjs test    run the backend test suite
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'ml-service')
const win = process.platform === 'win32'
const venvPython = path.join(root, '.venv', win ? 'Scripts' : 'bin', win ? 'python.exe' : 'python')
const run = (cmd, args) => {
  const r = spawnSync(cmd, args, { cwd: root, stdio: 'inherit' })
  process.exit(r.status ?? 1)
}

const task = process.argv[2] || 'dev'
if (task === 'setup') {
  const sys = win ? 'python' : 'python3'
  if (!existsSync(venvPython)) {
    const r = spawnSync(sys, ['-m', 'venv', '.venv'], { cwd: root, stdio: 'inherit' })
    if (r.status) process.exit(r.status)
  }
  run(venvPython, ['-m', 'pip', 'install', '-r', 'requirements-dev.txt'])
}
if (!existsSync(venvPython)) {
  console.error('ml-service/.venv not found. Run `pnpm ml:setup` first.')
  process.exit(1)
}
if (task === 'dev') run(venvPython, ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '127.0.0.1', '--port', process.env.ML_PORT || '8000'])
else if (task === 'test') run(venvPython, ['-m', 'pytest', '-q', '-p', 'no:warnings'])
else if (task === 'study') run(venvPython, ['-m', 'app.study.runner', ...process.argv.slice(3)])
else { console.error(`Unknown task: ${task}`); process.exit(1) }
