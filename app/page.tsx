import SocConsole from '@/components/soc-console'
import { analystProfile } from '@/lib/profile'

export const dynamic = 'force-dynamic'

export default function Page() {
  return <SocConsole profile={analystProfile()} />
}
