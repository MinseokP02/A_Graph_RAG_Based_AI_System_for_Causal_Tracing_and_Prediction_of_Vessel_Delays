import { useMemo, useState, type FormEvent } from 'react'
import {
  AlertCircle,
  BrainCircuit,
  CalendarDays,
  CloudSun,
  Database,
  Gauge,
  Search,
  Ship,
  Trash2
} from 'lucide-react'

import {
  askPortMIS,
  type PortMISAskResponse,
  type PortMISRecord,
  type PortMISWeather
} from '@/api/lightrag'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table'
import { errorMessage } from '@/lib/utils'
import AisTrackMap from '@/features/AisTrackMap'

type QueryLog = {
  id: string
  requestedAt: string
  question: string
  status: string
  answer: string
  warnings: string[]
  records: number
}

const LOG_STORAGE_KEY = 'portmis-dashboard-query-logs'
const DELAY_THRESHOLD_MINUTES = 60

const readLogs = (): QueryLog[] => {
  try {
    const value = JSON.parse(localStorage.getItem(LOG_STORAGE_KEY) || '[]')
    return Array.isArray(value) ? value.slice(0, 50) : []
  } catch {
    return []
  }
}

const valueOrDash = (value: string | number | null | undefined) =>
  value === null || value === undefined || value === '' ? '미기록' : String(value)

const formatWeather = (weather?: PortMISWeather[]) => {
  const item = weather?.[0]
  if (!item) return '기상 관측 미기록'
  return [
    `기온 ${valueOrDash(item.temperatureC)} ℃`,
    `풍속 ${valueOrDash(item.windSpeedMs)} m/s`,
    `강우 ${valueOrDash(item.rainMm)} mm`
  ].join(' · ')
}

const delayAnalysis = (record: PortMISRecord) => {
  if (record.isDepartureDelayed !== true && record.departureDelayMinutes == null) {
    return '지연 여부 또는 지연 시간이 기록되지 않아 분석할 수 없습니다.'
  }
  const observations: string[] = []
  const weather = record.departureWeather?.[0]
  if (weather?.windSpeedMs != null) observations.push(`풍속 ${weather.windSpeedMs} m/s`)
  if (weather?.rainMm != null) observations.push(`강우 ${weather.rainMm} mm`)
  if ((record.controlEventCount || 0) > 0) observations.push(`관제 이벤트 ${record.controlEventCount}건`)
  if (record.berths?.[0]?.name) observations.push(`계선시설 ${record.berths[0].name}`)
  const context = observations.length ? `${observations.join(', ')}이 함께 관측되었습니다. ` : ''
  return `${context}공식 지연 원인은 기록되어 있지 않아 인과관계로 단정할 수 없습니다.`
}

const makeId = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`

const extractDateFromQuestion = (question: string) => {
  const numeric = question.match(/(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)/)
  const korean = question.match(/(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일/)
  const match = numeric || korean
  if (!match) return ''
  const [, year, month, day] = match
  return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
}

export default function PortMisDashboard() {
  const [question, setQuestion] = useState('선박의 출항 지연 현황과 가능한 지연 원인을 분석해줘')
  const [vesselName, setVesselName] = useState('')
  const [callSign, setCallSign] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [result, setResult] = useState<PortMISAskResponse | null>(null)
  const [logs, setLogs] = useState<QueryLog[]>(readLogs)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const forecastRequested = useMemo(() => question.includes('예측'), [question])
  const records = useMemo(
    () => result?.evidence?.flatMap((item) => (item.kind === 'graph' && item.record ? [item.record] : [])) || [],
    [result]
  )
  const summary = result?.graph?.summary || {}
  const forecast = result?.forecast
  const prediction = forecast?.predictions?.[0]

  const saveLogs = (next: QueryLog[]) => {
    const limited = next.slice(0, 50)
    setLogs(limited)
    localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(limited))
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!question.trim() || loading) return
    const naturalLanguageDate = extractDateFromQuestion(question)
    if (forecastRequested && !startDate && !naturalLanguageDate) {
      setError('예측 질문이 감지되었습니다. 예측할 날짜를 선택하거나 질문에 YYYY-MM-DD로 입력해 주세요.')
      return
    }
    setLoading(true)
    setError('')

    try {
      const targetDate = forecastRequested ? startDate || naturalLanguageDate : undefined
      const response = await askPortMIS({
        question: question.trim(),
        mode: 'neo4j_mix',
        use_documents: true,
        max_records: 25,
        forecast_steps: 1,
        delay_threshold_minutes: DELAY_THRESHOLD_MINUTES,
        plan: {
          intent: forecastRequested ? 'delay_forecast' : 'delays',
          date_basis: 'departure',
          ...(vesselName.trim() ? { vessel_name: vesselName.trim() } : {}),
          ...(callSign.trim() ? { call_sign: callSign.trim() } : {}),
          ...(targetDate ? { start_date: targetDate, end_date: targetDate } : {}),
          ...(!forecastRequested && startDate ? { start_date: startDate } : {}),
          ...(!forecastRequested && endDate ? { end_date: endDate } : {})
        }
      })
      setResult(response)
      saveLogs([
        {
          id: makeId(),
          requestedAt: new Date().toISOString(),
          question: question.trim(),
          status: response.status,
          answer: response.answer,
          warnings: response.warnings || [],
          records: response.graph?.returned_records || 0
        },
        ...logs
      ])
    } catch (requestError) {
      const message = errorMessage(requestError)
      setError(message)
      saveLogs([
        {
          id: makeId(),
          requestedAt: new Date().toISOString(),
          question: question.trim(),
          status: 'error',
          answer: message,
          warnings: [],
          records: 0
        },
        ...logs
      ])
    } finally {
      setLoading(false)
    }
  }

  const clearLogs = () => {
    setLogs([])
    localStorage.removeItem(LOG_STORAGE_KEY)
  }

  return (
    <div className="min-h-full bg-gradient-to-br from-slate-50 via-background to-emerald-50/40 p-4 dark:from-slate-950 dark:to-emerald-950/20 md:p-6">
      <div className="mx-auto flex max-w-[1800px] flex-col gap-5">
        <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-600 dark:text-emerald-400">
              <Ship className="size-4" /> PORT OPERATIONS INTELLIGENCE
            </div>
            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Port-MIS 운항 분석 대시보드</h1>
            <p className="mt-1 text-sm text-muted-foreground">선박, 출항 기상, 지연 원인과 TFT 예측을 표 형식으로 확인합니다.</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Database className="size-4" /> Neo4j · LightRAG · TFT
          </div>
        </div>

        <Card className={forecastRequested ? 'border-blue-300 dark:border-blue-800' : 'border-emerald-200/70 dark:border-emerald-900'}>
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2 text-base">
              {forecastRequested ? <BrainCircuit className="size-4" /> : <Search className="size-4" />}
              {forecastRequested ? 'TFT 지연 예측 조건' : '조회 조건'}
            </CardTitle>
            <CardDescription>
              {forecastRequested
                ? "질문에서 '예측'을 감지했습니다. 날짜를 선택하거나 질문에 날짜를 입력하면 해당 날짜의 지연을 분석합니다."
                : '기간과 선박 조건으로 지연 기록과 원인 근거를 조회합니다.'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
              <Input value={vesselName} onChange={(e) => setVesselName(e.target.value)} placeholder="선박명" aria-label="선박명" />
              <Input value={callSign} onChange={(e) => setCallSign(e.target.value)} placeholder="호출부호" aria-label="호출부호" />
              <label className="space-y-1 text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><CalendarDays className="size-3" /> {forecastRequested ? '예측 날짜 *' : '시작일'}</span>
                <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} aria-label={forecastRequested ? '예측 날짜' : '시작일'} />
              </label>
              {!forecastRequested && (
                <label className="space-y-1 text-xs text-muted-foreground">
                  <span>종료일</span>
                  <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} aria-label="종료일" />
                </label>
              )}
              <Input
                className={forecastRequested ? 'md:col-span-2 xl:col-span-2' : 'md:col-span-2'}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="예: 2026-11-03 TEST 선박의 지연 예측"
                aria-label="분석 질문"
              />
              <Button type="submit" disabled={loading} className="md:col-span-2 xl:col-span-6 xl:justify-self-end">
                {forecastRequested ? <BrainCircuit className="size-4" /> : <Search className="size-4" />}
                {loading ? '분석 중...' : forecastRequested ? 'TFT 예측 실행' : '조회 및 분석'}
              </Button>
            </form>
            {error && (
              <div className="mt-3 flex items-center gap-2 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
                <AlertCircle className="size-4" /> {error}
              </div>
            )}
          </CardContent>
        </Card>

        <AisTrackMap />

        {(forecastRequested || forecast) && (
          <Card className="border-blue-200 dark:border-blue-900">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base"><Gauge className="size-4" /> TFT 지연 예측 결과</CardTitle>
              <CardDescription>P10/P50/P90은 예측 분위수이며, 지연 확률은 분위수 보간 추정치입니다.</CardDescription>
            </CardHeader>
            <CardContent>
              {forecast?.status === 'ok' && prediction ? (
                <div className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="rounded-lg bg-blue-50 p-4 dark:bg-blue-950/30"><div className="text-xs text-muted-foreground">예상 지연(P50)</div><div className="mt-1 text-2xl font-bold text-blue-700 dark:text-blue-300">{prediction.p50.toFixed(1)}분</div></div>
                    <div className="rounded-lg bg-amber-50 p-4 dark:bg-amber-950/30"><div className="text-xs text-muted-foreground">{forecast.delay_threshold_minutes ?? DELAY_THRESHOLD_MINUTES}분 이상 지연</div><div className="mt-1 text-2xl font-bold text-amber-700 dark:text-amber-300">{prediction.delay_probability_percent == null ? '산출 불가' : `${prediction.delay_probability_percent.toFixed(1)}%`}</div></div>
                    <div className="rounded-lg bg-slate-100 p-4 dark:bg-slate-900"><div className="text-xs text-muted-foreground">예측 범위(P10~P90)</div><div className="mt-1 text-2xl font-bold">{prediction.p10.toFixed(1)}~{prediction.p90.toFixed(1)}분</div></div>
                  </div>
                  <Table>
                    <TableHeader><TableRow><TableHead>예측 날짜</TableHead><TableHead>모델</TableHead><TableHead>AIS 관측 수</TableHead><TableHead>최근 속력</TableHead><TableHead>최근 위치</TableHead><TableHead>총톤수</TableHead><TableHead>화물량</TableHead></TableRow></TableHeader>
                    <TableBody><TableRow><TableCell>{new Date(prediction.timestamp).toLocaleString('ko-KR')}</TableCell><TableCell>{valueOrDash(forecast.model_version)}</TableCell><TableCell>{valueOrDash(forecast.basis?.ais_observation_count)}</TableCell><TableCell>{forecast.basis?.latest_sog_knots == null ? '미기록' : `${forecast.basis.latest_sog_knots} kn`}</TableCell><TableCell>{forecast.basis?.latest_latitude == null ? '미기록' : `${forecast.basis.latest_latitude}, ${forecast.basis.latest_longitude}`}</TableCell><TableCell>{valueOrDash(forecast.basis?.gross_tonnage)}</TableCell><TableCell>{valueOrDash(forecast.basis?.cargo_ton)}</TableCell></TableRow></TableBody>
                  </Table>
                  <p className="text-xs text-muted-foreground">확률은 학습 데이터의 실제 발생률이 아니라 TFT 분위수 사이를 선형 보간한 10~90% 범위의 추정치입니다.</p>
                </div>
              ) : (
                <div className="flex items-center gap-2 rounded-md bg-slate-100 p-4 text-sm dark:bg-slate-900">
                  <AlertCircle className="size-4" />
                  {result?.status === 'clarification_required'
                    ? result.answer
                    : forecast
                      ? `현재 예측 상태: ${forecast.status}. ${result?.answer || ''}`
                      : '선박과 예측 날짜를 입력한 후 TFT 예측을 실행해 주세요.'}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-base">Port-MIS 통계</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <TableHeader><TableRow><TableHead>신고 상세</TableHead><TableHead>고유 선박</TableHead><TableHead>지연값 보유</TableHead><TableHead>체류값 보유</TableHead><TableHead>지연 기록</TableHead></TableRow></TableHeader>
              <TableBody><TableRow className="text-lg font-semibold"><TableCell>{summary.matchingRecords ?? '-'}</TableCell><TableCell>{summary.vessels ?? '-'}</TableCell><TableCell>{summary.knownDelayRecords ?? '-'}</TableCell><TableCell>{summary.knownStayRecords ?? '-'}</TableCell><TableCell className="text-amber-600">{summary.delayedRecords ?? '-'}</TableCell></TableRow></TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base"><CloudSun className="size-4" /> 선박 상세 및 지연 원인 분석</CardTitle>
            <CardDescription>기상과 관제 기록은 관련 정황이며 공식 원인이나 인과관계로 단정하지 않습니다.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader><TableRow><TableHead>선박명</TableHead><TableHead>입출항</TableHead><TableHead>지연</TableHead><TableHead>기상</TableHead><TableHead className="min-w-[320px]">지연 원인 분석</TableHead></TableRow></TableHeader>
              <TableBody>
                {records.length ? records.map((record) => (
                  <TableRow key={record.recordId}>
                    <TableCell><div className="font-medium">{valueOrDash(record.vesselName)}</div><div className="text-xs text-muted-foreground">호출부호 {valueOrDash(record.callSign)}</div></TableCell>
                    <TableCell className="whitespace-nowrap text-xs"><div>입항 {valueOrDash(record.arrivalTime)}</div><div>예정 {valueOrDash(record.plannedDepartureTime)}</div><div>출항 {valueOrDash(record.departureTime)}</div></TableCell>
                    <TableCell className={record.isDepartureDelayed ? 'font-semibold text-amber-600' : ''}>{record.departureDelayMinutes == null ? '미기록' : `${record.departureDelayMinutes}분`}</TableCell>
                    <TableCell className="min-w-[240px] text-xs">{formatWeather(record.departureWeather)}</TableCell>
                    <TableCell className="text-xs leading-5">{delayAnalysis(record)}</TableCell>
                  </TableRow>
                )) : <TableRow><TableCell colSpan={5} className="h-24 text-center text-muted-foreground">조회 조건을 입력하고 분석을 실행해 주세요.</TableCell></TableRow>}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
            <div><CardTitle className="text-base">Query 응답 로그</CardTitle><CardDescription>최근 50개 요청을 브라우저에 보관합니다.</CardDescription></div>
            <Button variant="ghost" size="sm" onClick={clearLogs} disabled={!logs.length}><Trash2 className="size-4" /> 로그 삭제</Button>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader><TableRow><TableHead>시각</TableHead><TableHead>상태</TableHead><TableHead>Query</TableHead><TableHead>건수</TableHead><TableHead className="min-w-[520px]">응답 / 경고</TableHead></TableRow></TableHeader>
              <TableBody>
                {logs.length ? logs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell className="whitespace-nowrap text-xs">{new Date(log.requestedAt).toLocaleString('ko-KR')}</TableCell>
                    <TableCell className={log.status === 'error' ? 'font-medium text-red-600' : 'font-medium text-emerald-600'}>{log.status}</TableCell>
                    <TableCell className="max-w-[280px] align-top">{log.question}</TableCell>
                    <TableCell>{log.records}</TableCell>
                    <TableCell className="align-top"><div className="whitespace-pre-wrap text-xs leading-5">{log.answer}</div>{log.warnings.length > 0 && <div className="mt-2 text-xs text-amber-600">경고: {log.warnings.join(' · ')}</div>}</TableCell>
                  </TableRow>
                )) : <TableRow><TableCell colSpan={5} className="h-20 text-center text-muted-foreground">아직 Query 로그가 없습니다.</TableCell></TableRow>}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
