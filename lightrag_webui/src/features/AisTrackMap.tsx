import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from 'react-leaflet'
import { Map, Radio, RefreshCw, Ship } from 'lucide-react'
import 'leaflet/dist/leaflet.css'

import { getAisTracks, type AisTrackedVessel } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { errorMessage } from '@/lib/utils'

const colorForMmsi = (mmsi: string) => {
  let hash = 0
  for (const value of mmsi) hash = (hash * 31 + value.charCodeAt(0)) % 360
  return `hsl(${hash} 72% 44%)`
}

function FitTracks({ vessels }: { vessels: AisTrackedVessel[] }) {
  const map = useMap()
  const positions = useMemo(
    () => vessels.flatMap((vessel) => vessel.points.map((point) => [point.latitude, point.longitude] as [number, number])),
    [vessels]
  )
  useEffect(() => {
    if (positions.length) map.fitBounds(positions, { padding: [30, 30], maxZoom: 11 })
  }, [map, positions])
  return null
}

export default function AisTrackMap() {
  const [hours, setHours] = useState(24)
  const [vessels, setVessels] = useState<AisTrackedVessel[]>([])
  const [summary, setSummary] = useState({ vessels: 0, points: 0, limited: false })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  const loadTracks = useCallback(async (signal?: AbortSignal) => {
    try {
      setError('')
      const response = await getAisTracks(hours, signal)
      setVessels(response.vessels)
      setSummary({ vessels: response.summary.vessels, points: response.summary.points, limited: Boolean(response.summary.limited) })
      setUpdatedAt(new Date())
    } catch (requestError) {
      if (!signal?.aborted) setError(errorMessage(requestError))
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [hours])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    void loadTracks(controller.signal)
    const interval = window.setInterval(() => void loadTracks(controller.signal), 15_000)
    return () => {
      controller.abort()
      window.clearInterval(interval)
    }
  }, [loadTracks])

  return (
    <Card className="overflow-hidden border-cyan-200/80 dark:border-cyan-900">
      <CardHeader className="gap-3 pb-3 md:flex-row md:items-center md:justify-between">
        <div>
          <CardTitle className="flex items-center gap-2 text-base"><Map className="size-4 text-cyan-600" /> 한국 연안 AIS 선박 흐름</CardTitle>
          <CardDescription className="mt-1">부산·인천을 포함한 한국 해역의 수신 위치를 MMSI별 이동 경로로 표시합니다.</CardDescription>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="flex items-center gap-1 rounded-full bg-cyan-50 px-3 py-1.5 text-cyan-800 dark:bg-cyan-950/40 dark:text-cyan-200"><Ship className="size-3.5" /> 선박 {summary.vessels}척 · 위치 {summary.points.toLocaleString()}건</span>
          <label className="flex items-center gap-2">
            <span className="text-muted-foreground">표시 기간</span>
            <select className="rounded-md border bg-background px-2 py-1.5" value={hours} onChange={(event) => setHours(Number(event.target.value))} aria-label="AIS 표시 기간">
              <option value={1}>최근 1시간</option><option value={6}>최근 6시간</option><option value={24}>최근 24시간</option><option value={72}>최근 3일</option><option value={168}>최근 7일</option>
            </select>
          </label>
          <Button variant="outline" size="sm" onClick={() => void loadTracks()} disabled={loading}><RefreshCw className={`size-3.5 ${loading ? 'animate-spin' : ''}`} /> 새로고침</Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative h-[520px] border-t">
          <MapContainer center={[36.2, 127.8]} zoom={7} className="h-full w-full" preferCanvas>
            <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <FitTracks vessels={vessels} />
            {vessels.map((vessel) => {
              const color = colorForMmsi(vessel.mmsi)
              const positions = vessel.points.map((point) => [point.latitude, point.longitude] as [number, number])
              return <Fragment key={vessel.mmsi}>
                {positions.length > 1 && <Polyline positions={positions} pathOptions={{ color, weight: 3, opacity: 0.75 }} />}
                <CircleMarker center={[vessel.latest.latitude, vessel.latest.longitude]} radius={7} pathOptions={{ color: '#ffffff', fillColor: color, fillOpacity: 0.95, weight: 2 }}>
                  <Popup><div className="min-w-48 space-y-1 text-sm"><div className="font-semibold">{vessel.shipName || `MMSI ${vessel.mmsi}`}</div><div>MMSI: {vessel.mmsi}</div><div>호출부호: {vessel.callSign || '미수신'}</div><div>속력: {vessel.latest.sog == null ? '미수신' : `${vessel.latest.sog} kn`}</div><div>침로: {vessel.latest.cog == null ? '미수신' : `${vessel.latest.cog}°`}</div><div>최근 수신: {new Date(vessel.latest.timestamp).toLocaleString('ko-KR')}</div></div></Popup>
                </CircleMarker>
              </Fragment>
            })}
          </MapContainer>
          {!loading && !vessels.length && !error && <div className="pointer-events-none absolute inset-x-0 top-4 z-[1000] mx-auto flex w-fit items-center gap-2 rounded-md bg-background/95 px-4 py-2 text-sm shadow"><Radio className="size-4" /> 선택 기간에 수신된 AIS 위치가 없습니다.</div>}
          {error && <div className="absolute inset-x-0 top-4 z-[1000] mx-auto w-fit rounded-md bg-red-50 px-4 py-2 text-sm text-red-700 shadow dark:bg-red-950 dark:text-red-200">AIS 지도 조회 실패: {error}</div>}
        </div>
        <div className="flex flex-wrap justify-between gap-2 border-t px-4 py-2 text-xs text-muted-foreground"><span>선은 시간순 항적, 원은 선박의 최신 수신 위치입니다. 15초마다 자동 갱신됩니다.</span><span>{summary.limited ? '표시 상한 10,000건 적용 · ' : ''}갱신 {updatedAt ? updatedAt.toLocaleTimeString('ko-KR') : '-'}</span></div>
      </CardContent>
    </Card>
  )
}
