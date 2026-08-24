import IncidentInvestigation from '@/components/incident-investigation'

export default async function IncidentPage({ params }: { params: Promise<{ id: string }> }) {
  await params
  return <IncidentInvestigation />
}
